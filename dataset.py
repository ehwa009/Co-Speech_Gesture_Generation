import numpy as np
import torch
import torch.utils.data

import constant

class TedDataset(torch.utils.data.Dataset):
    
    def __init__(self, src_word2idx, src_insts=None, tgt_insts=None):
        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))
        
        # create idx to word dictionary
        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        return len(self._src_insts)

    @property
    def scr_vocab_size(self):
        return len(self._src_word2idx)

    @property
    def src_word2idx(self):
        return self._src_word2idx

    @property
    def src_idx2word(self):
        return self._src_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._src_insts[idx], self._tgt_insts[idx]

####################################################################
#                         PREPROCESSING                            #
####################################################################
def paired_collate_fn(insts, opt):
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts = list(zip(*seq_pairs))

    max_src_len = max(len(inst) for inst in src_insts)
    max_tgt_len = max(len(inst) for inst in tgt_insts)

    frame_duration = opt.frame_duration
    sp_duration = max_src_len / opt.speech_sp

    pre_duration = opt.pre_motions * frame_duration
    motion_duration = opt.estimation_motions * frame_duration

    num_words_for_pre_motion = round(max_src_len * pre_duration / sp_duration)
    num_words_for_estimation = round(max_src_len * motion_duration / sp_duration)

    tgt_len = opt.pre_motions + opt.estimation_motions

    # padding src seq
    # batch_src_seq = np.array([ 
    #                 [constant.UNK] * num_words_for_pre_motion + inst + [constant.PAD] * (max_src_len - len(inst)) 
    #                 for inst in src_insts])
    batch_src_seq = np.array([ 
                        inst + [constant.PAD] * (max_src_len - len(inst)) 
                        for inst in src_insts])
    
    batch_tgt_seq = np.array([ 
                        np.vstack((inst, np.zeros((max_tgt_len - len(inst), inst.shape[1])))) 
                        for inst in tgt_insts])
    
    src_seqs = []
    src_lens = []
    tgt_seqs = []
    
    # dataset parsing
    parse_iter = batch_src_seq.shape[1] - num_words_for_pre_motion
    for i in range(0, parse_iter, num_words_for_estimation):
        # get words chunk
        sample_seq = batch_src_seq[:, i:i + num_words_for_pre_motion + num_words_for_estimation]

        # add SOS and EOS token
        bos = np.zeros((sample_seq.shape[0], 1)) + constant.BOS
        eos = np.zeros((sample_seq.shape[0], 1)) + constant.EOS
        sample_seq = np.hstack((bos, sample_seq))
        sample_seq = np.hstack((sample_seq, eos))

        # count sequence length
        seq_l = []
        for seq in sample_seq:
            l = np.count_nonzero(seq)
            if l > 0:
                seq_l.append(l)
            else:
                seq_l.append(1)
        
        sample_pos = batch_tgt_seq[:, i:i+tgt_len]

        # transpose and append data
        
        # sample_seq = np.transpose(sample_seq, (1, 0))
        # sample_pos = np.transpose(sample_pos, (1, 0, 2))

        src_seqs.append(torch.LongTensor(sample_seq))
        tgt_seqs.append(torch.LongTensor(sample_pos))
        src_lens.append(seq_l)

    return src_seqs, src_lens, tgt_seqs


def collate_fn(insts, opt):
    src_seqs_list, src_lens_list, tgt_seqs_list = paired_collate_fn(insts, opt)

    src_pos_list = []
    for src in src_seqs_list:
        src_pos_list.append(
            torch.LongTensor(np.array([
                    [pos_i+1 if w_i != constant.PAD else 0
                        for pos_i, w_i in enumerate(inst)]
                            for inst in src])))
    tgt_pos_list = []
    for tgt in tgt_seqs_list:
            tgt_pos_list.append(
                torch.LongTensor(np.array([
                    [pos_i+1 if not(np.array_equal(w_i, np.array([constant.PAD] * w_i.shape[0]))) 
                            else 0
                                for pos_i, w_i in enumerate(inst)]
                                    for inst in tgt])))
                
    return zip(src_seqs_list, src_lens_list, src_pos_list, tgt_seqs_list, tgt_pos_list)