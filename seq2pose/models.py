import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

class EncoderRNN(nn.Module):

    def __init__(self, emb_matrix, input_size, hidden_size, bidirectional, n_layers=1, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout       
        
        # lookup table from pre-trained embedding matrix (glove)
        emb_matrix = torch.from_numpy(emb_matrix).float()  
        self.embedding = nn.Embedding.from_pretrained(emb_matrix)
        # do not update the embedding layer
        self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(self.embedding.embedding_dim, 
                        hidden_size, bidirectional=self.bidirectional, 
                        num_layers=n_layers, dropout=self.dropout)


    def forward(self, input_seqs, input_lengths, hidden=None, infer=False):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpacked, backed to padded

        if self.bidirectional:
            output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

        return output, hidden

class Attn(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.1,
                 discrete_representation=False):
        super().__init__()

        # define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.discrete_representation = discrete_representation

        # define embedding layer
        if self.discrete_representation:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.dropout = nn.Dropout(dropout)

        # calc input size
        if self.discrete_representation:
            input_size = hidden_size  # embedding size
        
        linear_input_size = input_size + hidden_size

        # define layers
        self.attn = Attn(hidden_size)
        self.pre_linear = nn.Sequential(
            nn.Linear(linear_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        # self.out = nn.Linear(hidden_size * 2, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def freeze_attn(self):
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(self, motion_input, last_hidden, encoder_outputs):
        '''
        :param motion_input:
            motion input for current time step, in shape [batch x dim]
        :param last_hidden:
            last hidden state of the decoder, in shape [layers x batch x hidden_size]
        :param encoder_outputs:
            encoder outputs in shape [steps x batch x hidden_size]
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        '''

        if '1.2.0' in torch.__version__:
            self.gru.flatten_parameters()

        if self.discrete_representation:
            word_embedded = self.embedding(motion_input).view(1, motion_input.size(0), -1)  # [1 x B x embedding_dim]
            motion_input = self.dropout(word_embedded)
        else:
            motion_input = motion_input.view(1, motion_input.size(0), -1)  # [1 x batch x dim]

        # attention
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)  # [batch x 1 x T]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch x 1 x attn_size]
        context = context.transpose(0, 1)  # [1 x batch x attn_size]

        # make input vec
        rnn_input = torch.cat((motion_input, context), 2)  # [1 x batch x (dim + attn_size)]

        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        rnn_input = rnn_input.unsqueeze(0)

        # rnn
        output, hidden = self.gru(rnn_input, last_hidden)

        # post-fc
        output = output.squeeze(0)  # [1 x batch x hidden_size] -> [batch x hidden_size]
        output = self.out(output)

        return output, hidden, attn_weights


class Seq2Pose(nn.Module):

    def __init__(self, word_emb, batch_size, hidden_size, bidirectional, n_layers,
                dropout, out_dim):
        super().__init__()

        self.encoder = EncoderRNN(
                       emb_matrix=word_emb,
                       input_size=word_emb.shape[1],
                       hidden_size=hidden_size,
                       bidirectional=bidirectional,
                       n_layers=n_layers,
                       dropout=dropout)

        self.decoder = BahdanauAttnDecoderRNN(
                        input_size=out_dim,
                        hidden_size=hidden_size,
                        output_size=out_dim,
                        n_layers=n_layers,
                        dropout=dropout)

    def forward(self, opt, src_seq, src_len, tgt_seq, device):
        enc_out, enc_hid = self.encoder(src_seq, src_len)
        dec_hid = enc_hid[:self.decoder.n_layers]
        all_decoder_outputs = torch.zeros(tgt_seq.size(0), 
                                            tgt_seq.size(1), 
                                            tgt_seq.size(2)).to(device) 
        dec_in = tgt_seq[0].float()
        all_decoder_outputs[0] = dec_in
        
        use_teacher_forcing = True if random.random() < opt.tf_ratio else False # set teacher forcing ratio
        if use_teacher_forcing:
            for di in range(1, len(tgt_seq)):
                dec_out, dec_hid, _ = self.decoder(dec_in,
                                                    dec_hid,
                                                    enc_out)
                all_decoder_outputs[di] = dec_out
                dec_in = tgt_seq[di].float()
        else:
            for di in range(1, len(tgt_seq)):
                dec_out, dec_hid, _ = self.decoder(dec_in, 
                                                    dec_hid,
                                                    enc_out)
                all_decoder_outputs[di] = dec_out
                dec_in = dec_out.float()

        suc_p = all_decoder_outputs[-opt.estimation_motions:].transpose(0,1).float()
        ans_p = tgt_seq[-opt.estimation_motions:].transpose(0,1).float()

        return suc_p, ans_p
        


