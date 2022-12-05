import torch.nn as nn
import torch
from pfrnns import PFLSTMCell, PFGRUCell
import numpy as np

def conv(in_channels, out_channels, kernel_size=3, stride=1,
        dropout=0.0):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            )

class AudioLocalizer(nn.Module):

    def __init__(self, args):
        super(AudioLocalizer, self).__init__()
        self.num_particles = args.num_particles
        self.hidden_dim = args.h
        self.img_size = args.img_size
        self.chan_size = args.chan_size
        self.obs_emb = args.emb_obs
        self.act_emb = args.emb_act
        self.dropout_rate = args.dropout
        self.map_emb = args.map_emb
        total_emb = 4*self.obs_emb + 4*self.act_emb

        self.rnn = PFLSTMCell(self.num_particles, total_emb,
                self.hidden_dim, 32, 32, args.resamp_alpha)

        self.hidden2label = nn.Linear(self.hidden_dim, 2)

        self.conv1 = conv(self.chan_size, 16, kernel_size=5, stride=2, dropout=0.2)
        self.conv2 = conv(16, 32, kernel_size=3, stride=1, dropout=0.2)
        self.conv3 = conv(32, 8, kernel_size=3, stride=1, dropout=0)
        fake_map = torch.zeros(1, self.chan_size, self.img_size[0], self.img_size[1])
        fake_out = self.conv3(self.conv2(self.conv1(fake_map)))
        out_dim = np.prod(fake_out.shape).astype(int)

        self.map_embedding = nn.Linear(out_dim, self.map_emb)
        self.map2obs = nn.Linear(self.map_emb, self.obs_emb)
        self.map2act = nn.Linear(self.map_emb, self.act_emb)
        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        self.initialize = 'rand'
        self.args = args
        self.bp_length = args.bp_length

    def init_hidden(self, batch_size):
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        h0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        c0 = initializer(batch_size * self.num_particles, self.hidden_dim)
        p0 = torch.ones(batch_size * self.num_particles, 1) * np.log(1 / self.num_particles)
        hidden = (h0, c0, p0)

        return hidden

    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple([h.detach() for h in hidden])
        else:
            return hidden.detach()

    def proccess_picture(self, pic, bs):
        emb_pic = self.conv3(self.conv2(self.conv1(pic))).view(bs, -1)
        emb_pic = torch.relu(self.map_embedding(emb_pic))
        obs_map = torch.relu(self.map2obs(emb_pic))
        act_map = torch.relu(self.map2act(emb_pic))
        return obs_map, act_map
        

    def forward(self, gcc_mic, logmel_mic, intensity_foa, logmel_foa):
        batch_size = gcc_mic.size(0)
        obs_gcc_map, act_gcc_map = self.proccess_picture(gcc_mic, batch_size)
        obs_logmel_gcc_map, act_logmel_gcc_map = self.proccess_picture(logmel_mic, batch_size)
        obs_intensity_map, act_intensity_map = self.proccess_picture(intensity_foa, batch_size)
        obs_foa_map, act_foa_map = self.proccess_picture(logmel_foa, batch_size)

        embedding_obs = torch.cat((obs_gcc_map, obs_logmel_gcc_map, obs_intensity_map,obs_foa_map), dim=1)
        embedding_act = torch.cat((act_gcc_map, act_logmel_gcc_map, act_intensity_map,act_foa_map), dim=1)

        embedding = torch.cat((embedding_obs, embedding_act), dim=1).unsqueeze(1)
        print("embedding.shape", embedding.shape)

        # repeat the input if using the PF-RNN
        embedding = embedding.repeat(self.num_particles, 1,1)
        print("embedding.shape", embedding.shape)
        seq_len = embedding.size(1)
        hidden = self.init_hidden(batch_size)

        hidden_states = []
        probs = []

        for step in range(seq_len):
            hidden = self.rnn(embedding[:, step, :], hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[-1])

            # if step % self.bp_length == 0:
            #     hidden = self.detach_hidden(hidden)

        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = self.hnn_dropout(hidden_states)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.num_particles, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.num_particles, -1, self.hidden_dim])
        y = out_reshape * torch.exp(prob_reshape)
        y = torch.sum(y, dim=1)
        y = self.hidden2label(y)
        pf_labels = self.hidden2label(hidden_states)

        y_out = torch.sigmoid(y)
        
        pf_out = torch.sigmoid(pf_labels[:, :, :2])
    
        return y_out, pf_out

    def step(self, gcc_mic, logmel_mic, intensity_foa, logmel_foa, gt_pos, args):

        pred, particle_pred = self.forward(gcc_mic, logmel_mic, intensity_foa, logmel_foa)

        gt_e = gt_pos[:, :, 0]
        gt_theta = gt_pos[:, :, 1]
        gt_e_normalized = (gt_e - 40) / (80)
        gt_theta_normalized = (gt_e - 180) / (360)
        gt_normalized = torch.cat([gt_e_normalized, gt_theta_normalized], dim=2)

        batch_size = pred.size(1)
        sl = pred.size(0)
        bpdecay_params = np.exp(args.bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
        if torch.cuda.is_available():
            bpdecay_params = torch.FloatTensor(bpdecay_params).cuda()
        else:
            bpdecay_params = torch.FloatTensor(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        pred = pred.transpose(0, 1).contiguous()

        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params

        l2_e_loss = torch.sum(l2_pred_loss[:, :, :2])
        l2_theta_loss = torch.sum(l2_pred_loss[:, :, 2])
        l2_loss = l2_e_loss + args.h_weight * l2_theta_loss

        l1_xy_loss = torch.mean(l1_pred_loss[:, :, :2])
        l1_h_loss = torch.mean(l1_pred_loss[:, :, 2])
        l1_loss = 10*l1_xy_loss + args.h_weight * l1_h_loss

        pred_loss = args.l2_weight * l2_loss + args.l1_weight * l1_loss

        total_loss = pred_loss

        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(self.num_particles, 1, 1)
        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl, 3)
        l2_particle_loss = - y_prob_l2.mean(dim=0).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.num_particles, -1, sl, 3)
        l1_particle_loss = - y_prob_l1.mean(dim=0).log()

        xy_l2_particle_loss = torch.mean(l2_particle_loss[:, :, :2])
        h_l2_particle_loss = torch.mean(l2_particle_loss[:, :, 2])
        l2_particle_loss = xy_l2_particle_loss + args.h_weight * h_l2_particle_loss

        xy_l1_particle_loss = torch.mean(l1_particle_loss[:, :, :2])
        h_l1_particle_loss = torch.mean(l1_particle_loss[:, :, 2])
        l1_particle_loss = 10 * xy_l1_particle_loss + args.h_weight * h_l1_particle_loss

        belief_loss = args.l2_weight * l2_particle_loss + args.l1_weight * l1_particle_loss
        total_loss = total_loss + args.elbo_weight * belief_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1, :2] * self.map_size, gt_pos[:, -1, :2])

        particle_pred = particle_pred.view(self.num_particles, batch_size, sl, 3)

        return total_loss, loss_last, particle_pred


import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.seq_len = len(self.data[0])
        self.seq_num = len(self.data)
        self.labels = labels
        self.samp_seq_len = None

    def __len__(self):
        return self.seq_num

    def set_samp_seq_len(self, seq_len):
        self.samp_seq_len = seq_len

    def __getitem__(self, index):
        seq_idx = index % self.seq_num

        traj = self.data[seq_idx]
        lbl = self.labels[seq_idx]

        traj = torch.FloatTensor(traj)
        lbl = torch.FloatTensor(lbl)

        return (*traj, lbl)
