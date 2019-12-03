import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot, transforms
from collections import namedtuple
from plot import Plot
from seq2pose.models import Seq2Pose

import constant as Constant

import matplotlib.pyplot as plt
import math
import re
import time, sys, pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(encoder, decoder, input_words, pre_motion_seq, opt, data):
    # make sure encoder and decoder be evaluation mode
    
    # +2 for SOS and EOS
    input_length = [len(input_words) + 2]
    input_seq = np.zeros((input_length[0], 1)) # seq x batch
    # add EOS
    input_seq[0, 0] = Constant.EOS
    
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
    
    # encoding
    encoder_outputs, encoder_hidden = encoder(input_seq, input_length, None)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    target_length = opt.pre_motions + opt.estimation_motions
    motion_output = np.array([])
    attentions = torch.zeros((target_length, len(input_seq))) 
    
    # time step
    for t in range(target_length):
        if t < opt.pre_motions:
            decoder_input = pre_motion_seq[t].unsqueeze(0).to(device).float()
            decoder_output, decoder_hidden, attn_weight = decoder(decoder_input, 
                                                                decoder_hidden,
                                                                encoder_outputs)
        else:
            decoder_input = decoder_output
            decoder_output, decoder_hidden, attn_weight = decoder(decoder_input, 
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


def normalized_string(sentence):
    sentence = sentence.lower()
    text = re.sub(r'[-=+.,*:]', '', sentence)
    text = text.strip()

    return text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./processed_data/preprocessing.pickle')
    parser.add_argument('-checkpoint', default='./trained_model/seq2pos_tr_loss_160_-1.322.chkpt')

    arg = parser.parse_args()

    data = torch.load(arg.data)
    model_info = torch.load(arg.checkpoint)

    ############################################
    #               Prepare Model              #
    ############################################
    state = model_info['model']
    opt = model_info['settings']

    if opt.model == 'transformer':
        print('[INFO] Transformer model selected.')
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

    encoder = model.encoder
    decoder = model.decoder

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

        padded_words = ['<UNK>'] * num_words_for_pre_motion + words

        # output tuple to save all related information
        output_tuple = namedtuple('InferenceOutput', ['words', 'pre_motion_seq', 'out_motion', 'attention'])
        
        # previous motion seq
        pre_motion_seq = np.zeros((opt.pre_motions, data['pca'].n_components))

        # to store motion outputs
        outputs = []
        for i in range(0, len(padded_words) - num_words_for_pre_motion, num_words_for_estimation):
            sample_words = padded_words[i:i + num_words_for_pre_motion + num_words_for_estimation]
            
            with torch.no_grad():
                output, attention = inference(
                                        encoder=encoder,
                                        decoder=decoder,
                                        input_words=sample_words,
                                        pre_motion_seq=pre_motion_seq,
                                        opt=opt,
                                        data=data)
                outputs.append(output_tuple(sample_words, pre_motion_seq, output, attention))

                # set previous 10 motions as a next intput
                pre_motion_seq = np.asarray(output)[-opt.pre_motions:, :]
            
        return outputs


    # inference
    sentence = "look at the big world in front of you ,"
    # sentence = "look at the small world in front of me ,"
    # sentence = "but what you hold in your hand leaves a bloody trail"
    # sentence = "and the most staggering thing of all of this, to me"
    # sentence = '''One of the examples provided on the matplotlib example page is an animation of a double pendulum. 
    # #             This example operates by precomputing the pendulum position over 10 seconds, and then animating the results. 
    # #             I saw this and wondered if python would be fast enough to compute the dynamics on the fly. 
    # #             It turns out it is'''
    # sentence = "Witnesses told the Herald the brawl kicked off around 3pm and at one point a beer bottle was smashed over the head of a teen"
    
    words = normalized_string(sentence).split(' ')
    outputs = infer_from_words(words)

    poses = np.zeros((len(outputs)*20, 24)) # to store the outputs
    print("output pose frames: {}".format(poses.shape[0]))
    
    # we define offset to maximize gesture generated
    offset = 1.0

    start = 0
    for out in outputs:
        out_m = np.array(out.out_motion)
        out_m = out_m * offset
        for pose in out_m:
            pose[2] = 0. # force to delete in-rotation motions
            poses[start] = data['pca'].inverse_transform(pose)
            start += 1
    
    # plot class
    p = Plot((3, 10), (6, 13))
    # p = Plot((-20, -7), (-20, 10))
    anim = p.animate(poses, 100)
    anim.save("./figures/{}.mp4".format(sentence[:30]))
    # p.display_multi_poses(poses)
    # plt.show()


if __name__ == '__main__':
    main()
  