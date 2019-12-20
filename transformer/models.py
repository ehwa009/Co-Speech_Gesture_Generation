import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import constant as Constants
import random

from transformer.layers import EncoderLayer, DecoderLayer


class PostionalEncoding(nn.Module):

    def __init__ (self, d_hid, n_position=200):
        super(PostionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


# # test
# position_enc = PostionalEncoding(128, 50)
# pos_encoding = position_enc._buffers['pos_table'].squeeze(0)
# plt.pcolormesh(pos_encoding, cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0,128))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
# exit(-1)



def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    return _get_subsequent_mask(len_s, seq.device)


def _get_subsequent_mask(len_s, device):
    subsequent_mask = 1 - torch.triu(
        torch.ones((1, len_s, len_s), device=device, dtype=torch.unit8), diagonal=1)

    return subsequent_mask


class Encoder(nn.Module):

    def __init__(self, emb_matrix, n_src_vocab, d_word_vec, n_layers, n_head, 
                d_k, d_v, d_enc_model, d_inner, dropout=0.1, n_position=200):
        
        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        emb_matrix = torch.from_numpy(emb_matrix).float()
        self.src_word_emb = nn.Embedding.from_pretrained(emb_matrix, freeze=True)
        self.postion_enc = PostionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
                        EncoderLayer(d_enc_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                        for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_enc_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        # forward
        enc_output = self.dropout(self.postion_enc(self.src_word_emb(src_seq)))
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):

    def __init__(self, d_motion_vec, n_layers, n_head, 
                d_k, d_v, d_dec_model, d_inner, dropout=0.1, 
                n_position=200, output_size=10):
        super().__init__()
        
        self.position_enc = PostionalEncoding(d_motion_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.pre = nn.Sequential(
            nn.Linear(output_size, d_dec_model),
            nn.BatchNorm1d(d_dec_model),
            nn.ReLU(inplace=True)
        )
        self.layer_stack = nn.ModuleList([
                    DecoderLayer(d_dec_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                    for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_dec_model, eps=1e-6)
        self.out = nn.Linear(d_dec_model, output_size)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
        trg_seq = trg_seq.unsqueeze(1) # b x s x dim
        # forwrad
        dec_output = self.dropout(self.position_enc(trg_seq))
        dec_output = dec_output.view(1, dec_output.size(0), -1) # 1 x batch x dim
        dec_output = self.pre(dec_output.squeeze(0)).unsqueeze(1) # b x s x dim
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        dec_output = self.layer_norm(dec_output)
        dec_output = dec_output.squeeze(1)
        dec_output = self.out(dec_output)
            
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,
    


class Transformer(nn.Module):

    def __init__(
                self, emb_matrix,
                n_src_vocab, src_pad_idx, d_word_vec=300, d_enc_model=300, 
                d_dec_model=10, d_motion_vec=10, d_inner=2048, n_layers=6, n_head=8, 
                d_k=64, d_v=64, dropout=0.1, n_position=200):
        
        super().__init__()
        self.src_pad_idx = src_pad_idx
        
        self.encoder = Encoder(
                emb_matrix=emb_matrix, n_src_vocab=n_src_vocab, n_position=n_position, 
                d_word_vec=d_word_vec, d_enc_model=d_enc_model, d_inner=d_inner, 
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.decoder = Decoder(
                d_motion_vec=d_motion_vec, d_k=d_k, d_v=d_v, 
                n_layers=n_layers, n_head=n_head, d_dec_model=d_dec_model, 
                d_inner=d_inner, dropout=dropout)

    def forward(self, opt, src_seq, trg_seq, device):
        # forward encoder
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)
        
        trg_seq = trg_seq.permute(1, 0, 2) # s x b x 10
        # to save generated poses
        all_decoder_outputs = torch.zeros(trg_seq.size(0), 
                                            trg_seq.size(1), 
                                            trg_seq.size(2)).to(device)
        dec_in = trg_seq[0].float()
        all_decoder_outputs[0] = dec_in
        
        # foward decoder
        use_teacher_forcing = True if random.random() < opt.tf_ratio else False # set teacher forcing ratio
        if use_teacher_forcing:
            for di in range(1, len(trg_seq)):
                dec_output, *_ = self.decoder(dec_in, None, enc_output, src_mask)
                all_decoder_outputs[di] = dec_output
                dec_in = trg_seq[di].float()
        else:
            for di in range(1, len(trg_seq)):
                dec_output, *_ = self.decoder(dec_in, None, enc_output, src_mask)
                all_decoder_outputs[di] = dec_output
                dec_in = dec_output.float()

        suc_p = all_decoder_outputs[-opt.estimation_motions:].transpose(0,1).float()
        ans_p = trg_seq[-opt.estimation_motions:].transpose(0,1).float()

        return suc_p, ans_p
