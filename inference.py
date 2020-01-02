import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import pyplot, transforms
from collections import namedtuple
from plot import Plot
from seq2pose.models import Seq2Pose
from transformer.models import Transformer, get_pad_mask, get_subsequent_mask

import constant as Constant

import matplotlib.pyplot as plt
import math
import random
import re
import time, sys, pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(model, input_words, pre_motion_seq, opt, data):
    # make sure encoder and decoder be evaluation mode
    
    # +2 for SOS and EOS
    input_length = [len(input_words) + 2]
    input_seq = np.zeros((input_length[0], 1)) # seq x batch
    # add EOS
    input_seq[0, 0] = Constant.BOS
    
    for i, word in enumerate(input_words):
        try:
            word_idx = data['dict'][word]
        except KeyError:
            word_idx = Constant.UNK
        input_seq[i + 1, 0] = word_idx
    
    # add EOS
    input_seq[input_seq.shape[0] - 1, 0] = Constant.EOS
    
    input_seq = torch.from_numpy(input_seq).long().to(device)
    pre_motion_seq = torch.from_numpy(pre_motion_seq).float().to(device)
    
    if opt.model == 'seq2pos':
        # encoding
        encoder_outputs, encoder_hidden = model.encoder(input_seq, input_length, None)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        
        target_length = opt.pre_motions + opt.estimation_motions
        motion_output = np.array([])
        attentions = torch.zeros((target_length, len(input_seq))) 
        
        # time step
        for t in range(target_length):
            if t < opt.pre_motions:
                decoder_input = pre_motion_seq[t].unsqueeze(0).to(device).float()
                decoder_output, decoder_hidden, attn_weight = model.decoder(decoder_input, 
                                                                    decoder_hidden,
                                                                    encoder_outputs)
            else:
                decoder_input = decoder_output
                decoder_output, decoder_hidden, attn_weight = model.decoder(decoder_input, 
                                                            decoder_hidden, 
                                                            encoder_outputs)
                decoder_input = decoder_output.float()
                
                if t == opt.pre_motions:
                    motion_output = decoder_output.data.cpu().numpy()
                else:
                    motion_output = np.vstack((motion_output, decoder_output.data.cpu().numpy()))
            
            if attn_weight is not None:
                attentions[t] = attn_weight.data
        
        return motion_output, attentions
    
    elif opt.model == 'transformer':
        input_seq = input_seq.transpose(0, 1)
        pre_motion_seq = pre_motion_seq.view(1, -1, 10) # b x s x dim
        pre_motion_seq = torch.zeros(1,30,10).to(device)

        target_length = opt.pre_motions + opt.estimation_motions
        
        # forward encoder
        input_mask = get_pad_mask(input_seq, opt.src_pad_idx)
        enc_output, *_ = model.encoder(input_seq, input_mask)
        
        # forward pad dense layer
        enc_output = model.pad_linear(enc_output)

        dec_output = model.decoder(pre_motion_seq, None, enc_output, input_mask)
        motion_output = dec_output[0].squeeze(0).data.cpu().numpy()

        # # forward decoder
        # for t in range(0, target_length, 10):
        #     if t < opt.pre_motions:
        #         dec_output, *_ = model.decoder(pre_motion_seq, None, enc_output, input_mask)
        #     else:
        #         decoder_input = dec_output
        #         dec_output, *_ = model.decoder(decoder_input, None, enc_output, input_mask)
        #         decoder_input = dec_output.float()
                
        #         if t == opt.pre_motions:
        #             motion_output = dec_output.squeeze(0).data.cpu().numpy()
        #         else:
        #             motion_output = np.vstack((motion_output, dec_output.squeeze(0).data.cpu().numpy()))

        return motion_output, None


def _model_decode(decoder, trg_seq, enc_output, src_mask):
    dec_output, *_ = decoder(trg_seq, None, enc_output, src_mask)
    return dec_output


def _get_init_state(encoder, decoder, init_seq, src_seq, src_mask):
    enc_output, *_ = encoder(src_seq, src_mask)
    dec_output = _model_decode(decoder, init_seq, enc_output, src_mask)
    return enc_output, dec_output


def normalized_string(sentence):
    sentence = sentence.lower()
    text = re.sub(r'[-=+.,*:]', '', sentence)
    text = text.strip()
    return text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./processed_data/preprocessing.pickle')
    # parser.add_argument('-checkpoint', default='./trained_model/trained_model_seq2pos/seq2pos_tr_loss_570_-1.354.chkpt')
    parser.add_argument('-checkpoint', default='./trained_model/transformer.chkpt')
    parser.add_argument('-ground_truth', type=bool, default=False)
    parser.add_argument('-n_filter', type=int, default=3)

    arg = parser.parse_args()

    data = torch.load(arg.data)
    model_info = torch.load(arg.checkpoint)

    if arg.ground_truth:
        index = random.randrange(0, len(data['valid']['src']))
        sample_src = data['valid']['src'][index]
        sample_tgt = data['valid']['tgt'][index]
        poses = np.zeros((len(sample_tgt), 24)) # to store the outputs
        word = []
        for src in sample_src:
            for k, v in data['dict'].items():
                if src == v:
                    word.append(k)
        print('selected index: {}'.format(index))
        print('sample words: {} \n\t{}'.format(len(word), word))
        print('pose length: {}'.format(len(sample_tgt)))
        
        start = 0
        for pose in sample_tgt:
            poses[start] = data['pca'].inverse_transform(pose)
            start += 1
        p = Plot((-7, 7), (-7, 7))
        poses = pd.DataFrame(poses).rolling(arg.n_filter).mean()
        poses = np.array(poses)
        anim = p.animate(poses, 80)
        # p.save(anim, "./videos/groud_truth.mp4")
        plt.show()
        exit(-1)


    ############################################
    #               Prepare Model              #
    ############################################
    state = model_info['model']
    opt = model_info['settings']
    # opt.pre_motions = 5

    if opt.model == 'transformer':
        print('[INFO] Transformer model selected.')
        model = Transformer(
                emb_matrix=data['emb_tbl'],
                n_src_vocab=opt.scr_vocab_size,
                src_pad_idx = opt.src_pad_idx,
                trg_pad_idx = opt.trg_pad_idx,
                d_enc_model=opt.d_enc_model,
                d_dec_model=opt.d_dec_model,
                d_inner=opt.d_inner_hid,
                n_layers=opt.n_layers,
                d_k=opt.d_k,
                d_v=opt.d_v,
                n_head=opt.n_head,
                dropout=opt.dropout).to(device)
        # load trained state
        model.load_state_dict(state)
        # turn model into evaluation mode
        model.eval()
    
    elif opt.model == 'seq2pos':
        print('[INFO] Seq2Pos model selected.')
        model = Seq2Pose(
                word_emb=data['emb_tbl'], 
                batch_size=1, 
                hidden_size=opt.hidden_size, 
                n_enc_layers=opt.n_enc_layers,
                n_dec_layers=opt.n_dec_layers,
                bidirectional=opt.bidirectional,
                dropout=opt.dropout,
                out_dim = data['pca'].n_components).to(device)
        # load trained state
        model.load_state_dict(state)
        # turn model into evaluation mode
        model.eval()

    def infer_from_words(words, sp_duration=None):
        start = time.time()

        if sp_duration is None:
            # speech duration
            # assume average speech speed (150 wpm = 2.5 wps)
            sp_duration = len(words) / 2.5

        # prefix values
        # unit_dration = 0.08333 # seconds per frame (dataset has 12 fps)
        frame_duration = 1/12

        pre_duration = opt.pre_motions * frame_duration
        motion_duration = opt.estimation_motions * frame_duration

        num_words_for_pre_motion = round(len(words) * pre_duration / sp_duration)
        num_words_for_estimation = round(len(words) * motion_duration / sp_duration)

        padded_words = [Constant.UNK_WORD] * num_words_for_pre_motion + words

        # output tuple to save all related information
        output_tuple = namedtuple('InferenceOutput', ['words', 'pre_motion_seq', 'out_motion', 'attention'])
        
        # previous motion seq
        # pre_motion_seq = np.zeros((opt.pre_motions, data['pca'].n_components))
        pre_motion_seq = np.zeros((30, data['pca'].n_components))

        # to store motion outputs
        outputs = []
        for i in range(0, len(padded_words) - num_words_for_pre_motion, num_words_for_estimation):
            sample_words = padded_words[i:i + num_words_for_pre_motion + num_words_for_estimation]
            with torch.no_grad():
                output, attention = inference(
                                        model=model,
                                        input_words=sample_words,
                                        pre_motion_seq=pre_motion_seq,
                                        opt=opt,
                                        data=data)
                
                outputs.append(output_tuple(sample_words, pre_motion_seq, output, attention))
                # pre_motion_seq = np.asarray(output)[-opt.pre_motions:, :]
                pre_motion_seq = np.asarray(output)[:]
                
        return outputs


    # inference
    # sentence = "look at the big world in front of you ,"
    sentence = "look at the small world in front of me ,"
    # sentence = "but what you hold in your hand leaves a bloody trail"
    # sentence = "and the most staggering thing of all of this, to me"
    # sentence = '''and men in general are physically stronger of course there are many exceptions laughter but today we live in a vastly different world the person more likely to lead is not the physically stronger person it is the more creative person the more intelligent person the more innovative person and there are no hormones for those attributes a man is as likely as a woman to be intelligent to be creative to be innovative we have evolved but it seems to me that our ideas of gender had not evolved some weeks ago i walked into a lobby of one of the best nigerian hotels i thought about naming the hotel but i thought i probably shouldnt and a guard at the entrance stopped me and asked me annoying questions because their automatic assumption is that a nigerian female walking into a hotel alone is a sex worker'''
    # sentence = "Witnesses told the Herald the brawl kicked off around 3pm and at one point a beer bottle was smashed over the head of a teen"
    
    words = normalized_string(sentence).split(' ')
    outputs = infer_from_words(words)

    poses = np.zeros((len(outputs)*30, 24)) # to store the outputs
    print("output pose frames: {}".format(poses.shape[0]))
    
    # we define offset to maximize gesture generated
    offset = 1.0

    start = 0
    for out in outputs:
        out_m = np.array(out.out_motion)
        out_m = out_m * offset
        for pose in out_m:
            poses[start] = data['pca'].inverse_transform(pose)
            start += 1
    
    # save output poses
    # torch.save(poses, "./videos/output.pickle")
    # print("[INFO] output saved.")
    # exit(-1)

    # plot class
    p = Plot((-7, 7), (-7, 7))
    # mean average filtering
    poses = pd.DataFrame(poses).rolling(arg.n_filter).mean()
    poses = np.array(poses)
    anim = p.animate(poses, 100000)
    p.save(anim, "./videos/predict_test.mp4")
    # plt.show()


if __name__ == '__main__':
    main()
  