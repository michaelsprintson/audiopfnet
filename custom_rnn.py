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

class AudioRNN(nn.Module):

    def __init__(self, args):
        super(AudioRNN, self).__init__()
        self.hidden_dim = args.h
        self.img_size = args.img_size
        self.chan_size = args.chan_size
        self.obs_emb = args.emb_obs
        self.act_emb = args.emb_act
        self.dropout_rate = args.dropout
        self.map_emb = args.map_emb
        total_emb = 4*self.obs_emb + 4*self.act_emb

        self.rnn = torch.nn.LSTM(total_emb, self.hidden_dim, batch_first = True)
        
        # PFLSTMCell(self.num_particles, total_emb,
        #         self.hidden_dim, 32, 32, args.resamp_alpha)

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

        h0 = initializer(1, batch_size, self.hidden_dim)
        c0 = initializer(1, batch_size, self.hidden_dim)
        hidden = (h0, c0)

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

        embedding = torch.cat((embedding_obs, embedding_act), dim=1)
        print(embedding.shape)

        # repeat the input if using the PF-RNN
        hidden = self.init_hidden(batch_size)

        # hidden_states = []

        o, hidden = self.rnn(embedding, hidden)
        # hidden_states.append(hidden[0])

            # if step % self.bp_length == 0:
            #     hidden = self.detach_hidden(hidden)

        # hidden_states = torch.stack(hidden_states, dim=0)
        # hidden_states = self.hnn_dropout(hidden_states)

        y = self.hidden2label(hidden)
        # pf_labels = self.hidden2label(hidden_states)

        y_out = torch.sigmoid(y)
        print(y_out.shape)
        
        # pf_out = torch.sigmoid(pf_labels[:, :, :2])
    
        return y_out, None

    def step(self, gcc_mic, logmel_mic, intensity_foa, logmel_foa, gt_pos, args):

        pred, particle_pred = self.forward(gcc_mic, logmel_mic, intensity_foa, logmel_foa)

        gt_pos = gt_pos.unsqueeze(1)
        gt_pos = gt_pos.repeat(1, self.num_particles,1)
        gt_e = gt_pos[:, :, 0].unsqueeze(2)
        gt_theta = gt_pos[:, :, 1].unsqueeze(2)
        gt_e_normalized = (gt_e - 40) / (80)
        gt_theta_normalized = (gt_theta - 180) / (360)
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

        l2_e_loss = torch.sum(l2_pred_loss[:, :, 0])
        l2_theta_loss = torch.sum(l2_pred_loss[:, :, 1])
        l2_loss = l2_e_loss + l2_theta_loss

        l1_xy_loss = torch.mean(l1_pred_loss[:, :, 0])
        l1_h_loss = torch.mean(l1_pred_loss[:, :, 1])
        l1_loss = l1_xy_loss + l1_h_loss

        pred_loss = args.l2_weight * l2_loss + args.l1_weight * l1_loss

        total_loss = pred_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1, 0] * 40, gt_pos[:, -1, 0])
        particle_pred = particle_pred.view(self.num_particles, batch_size, sl, 2)

        return total_loss, loss_last, pred