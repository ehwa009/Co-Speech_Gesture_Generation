import argparse
import torch
import pickle
import math
import constant as Constants
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from plot import display_multi_poses, display_pose
from sklearn.decomposition import PCA
from sklearn import preprocessing

def loadpickle(path, data_size):
    '''
    load train and val dataset

    param:
        file path
        data_size
    return:
        data
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)     
    
    # retrict loaded data size
    if data_size < len(data):
        dataset = data[:data_size]
    else:
        dataset = data[:]
    
    return dataset


def get_data(data, sampling_rate):
    x_train = []
    y_train = []
    x_tmp = []
    y_tmp = []

    # the poses in the dataset has been captured with 12fps
    # sampling_rate = math.ceil(12 / sampling_rate)

    for data in data:
        if len(data['clips'])>0:
            for clip in data['clips']:              
                
                # get words in a clip
                words = clip['words']
                
                # assign temp word list
                sentence = []
                for word in words:
                    w = word[0] # get a word
                    if w != '':
                        sentence.append(w) # full sentence
                
                # add indexed words to x_train
                x_tmp.append(sentence)
                
                # add skeletons in a clip to y_train
                y_tmp.append(clip['skeletons'])
    
    pair_list = list(zip(x_tmp, y_tmp))
    
    for pair in pair_list:
        sentence = pair[0]
        poses = pair[1]
        tmp_poses = []
        dist1_list = []
        dist2_list = []
        for p in poses:
            if not(0 in p):
                p = np.array(p) * -1 # rotate whole pose
                p += 1500 # make it positive number
                tmp_poses.append(p)
                
        # sampling 10fps
        tmp_poses = tmp_poses[::sampling_rate] 
            
        # selecte dataset with below condition;
        #                   1. pose seq must be longer than word seq
        #                   2. word seq has more than 12 (6*2)
        if (2 * len(sentence) < len(tmp_poses)) and (len(sentence) > 6*2):
            x_train.append(sentence)
            y_train.append(tmp_poses)

    print('[INFO] dataset desc.')
    print("\tparis: {}".format(len(x_train)))
    
    print("\tmax seq in x: {}".format(len(max(x_train, key=len))))
    print("\tmin seq in x: {}".format(len(min(x_train, key=len))))

    print("\tmax seq in y: {}".format(len(max(y_train, key=len))))
    print("\tmin seq in y: {}\n".format(len(min(y_train, key=len))))

    return x_train, y_train


def build_vocab_idx(word_insts, min_word_count):
    # word to index dictionary
    word2idx = {
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS,
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK,
            }

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[INFO] Original Vocabulary size: {}'.format(len(full_vocab)))

    word_count = {w: 0 for w in full_vocab}

    # count word frequency in the given dataset
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1
    
    ignored_word_count = 0
    
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx) # add word to dictioary with index
            else:
                ignored_word_count += 1

    print('[INFO] Trimmed vocabulary size: {}\neach with minum occurrence: {}'.format(len(word2idx), min_word_count))
    print('[INFO] Ignored word cound: {}'.format(ignored_word_count))

    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' 
        note: 
            mapping word to idx seq
            return unk seq if there is unknown seq
    '''
    return [ [word2idx.get(w, Constants.UNK) for w in s] for s in word_insts ]


def run_PCA_train_tgt(tgt_insts, lengths, n_components):
    pca = PCA(n_components=n_components)
    pca_tgt = pca.fit_transform(tgt_insts)

    ori_tgt = []
    # initial index of expanded pose array
    start = 0
    for l in lengths:
        # create empty array to store pca poses
        pca_skel = np.zeros((1, n_components))
        sel_p = pca_tgt[start:start+l]
        for i in range(sel_p.shape[0]):
            sel_p[i][2] = 0.00
        # stack
        ori_tgt.append(sel_p)
        # change index
        start = l
    
    return pca, ori_tgt


def run_PCA_val_tgt(pca, tgt_insts, lengths, n_components):
    pca_tgt = pca.transform(tgt_insts)
    ori_tgt = []
    # initial index of expanded pose array
    start = 0
    for l in lengths:
        # create empty array to store pca poses
        pca_skel = np.zeros((1, n_components))
        sel_p = pca_tgt[start:start+l]
        for i in range(sel_p.shape[0]):
            sel_p[i][2] = 0.00
        # stack
        ori_tgt.append(sel_p)
        # change index
        start = l
    
    return ori_tgt


def tgt_insts_normalize(tgt_insts):
    '''
    param:
        motion inputs list
        motion inputs list and normalize the values,
        noramlizer - min_max etc
    return:
        expanded normalized motion list, 
        motion lengths list
    '''

    def get_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def get_theta(var_x, var_y, fix_x, fix_y):
        return math.atan2(var_y - fix_y, var_x - fix_x)

    def get_new_cor(theta, dist, point):
        return dist * np.array([math.cos(theta), 
                                math.sin(theta)]) + np.array([point[0], point[1]])

    def length_norm(var_x, var_y, fix_x, fix_y, expanded_len):
        angle = get_theta(var_x, var_y, fix_x, fix_y)
        new_cor = get_new_cor(
                        angle,
                        expanded_len,
                        [fix_x, fix_y])
        # get change ratio between original dist and expected dist
        ratio = expanded_len / get_distance(var_x, var_y, fix_x, fix_y)
        # print("old cor: {:0.2f}, {:0.2f}".format(var_x, var_y))
        # print("new cor: {:0.2f}, {:0.2f}".format(new_cor[0], new_cor[1]))
        return new_cor, ratio

    # ------------------- Fitering poses --------------------- #
    tmp = []
    length = []
    # expand poses list
    for pose in tgt_insts:
        p_count = 0
        for p in pose:
            if 1.3*get_distance(p[6], p[7], p[3], p[4]) >= get_distance(p[0], p[1], p[3], p[4]) and \
                1.3*get_distance(p[15], p[16], p[3], p[4]) >= get_distance(p[0], p[1], p[3], p[4]):
                if get_distance(p[6], p[7], p[3], p[4]) < 1.3*get_distance(p[15], p[16], p[3], p[4]) and \
                    1.3*get_distance(p[6], p[7], p[3], p[4]) > get_distance(p[15], p[16], p[3], p[4]):
                    if (p[6] > p[3]) and (p[15] < p[3]): # rotated motion
                        if p[1] > p[4]:
                            tmp.append(p)
                            p_count += 1
        length.append(p_count)
    
    # convert list to np array
    tmp = np.array(tmp)
    # normalized with specific scale
    normalized = preprocessing.normalize(tmp, norm='l2') * 100
    print('[INFO] Filtered poses: {}'.format(len(normalized)))
    # save explanded pose pickle 
    print('[INFO] Save l2 noramlized pose.')
    torch.save(normalized, './processed_data/l2_norm.pickle')

    # get mean dist of each shoulders
    mean_val_pose = np.mean(normalized, axis=0)
    rig_sh_len_mean = get_distance(mean_val_pose[3], mean_val_pose[4], mean_val_pose[6], mean_val_pose[7])
    lef_sh_len_mean = get_distance(mean_val_pose[3], mean_val_pose[4], mean_val_pose[15], mean_val_pose[16])

    for pose in normalized:
        # ------------------- re-coordinate neck --------------------- #
        neck_diff_x = 0 - pose[3]
        neck_diff_y = 0 - pose[4]
        pose[3] = 0
        pose[4] = 0
        for i in range(len(pose)):
            if (i % 3 == 0) and not(i == 3): # x
                pose[i] += neck_diff_x
            elif (i % 3 == 1) and not(i == 4): # y
                pose[i] += neck_diff_y
    # save explanded pose pickle 
    print('[INFO] Save neck re-cordination.')
    torch.save(normalized, './processed_data/neck_loc.pickle')
    # exit(-1)
    for pose in normalized:
        # ------------------- normalize shoulder --------------------- #
        rig_new_cor, rig_ratio = length_norm(pose[6], pose[7], pose[3], pose[4], rig_sh_len_mean)
        lef_new_cor, lef_ratio = length_norm(pose[15], pose[16], pose[3], pose[4], lef_sh_len_mean)

        rig_diff_x = rig_new_cor[0] - pose[6]
        rig_diff_y = rig_new_cor[1] - pose[7]
        lef_diff_x = lef_new_cor[0] - pose[15]
        lef_diff_y = lef_new_cor[1] - pose[16]

        # shoudler re-loc
        pose[6] = rig_new_cor[0]
        pose[7] = rig_new_cor[1]
        pose[15] = lef_new_cor[0]
        pose[16] = lef_new_cor[1]

        # rest of cor re-loc
        pose[9] += rig_diff_x
        pose[10] += rig_diff_y

        pose[12] += rig_diff_x
        pose[13] += rig_diff_y

        pose[18] += lef_diff_x
        pose[19] += lef_diff_y

        pose[21] += lef_diff_x
        pose[22] += lef_diff_y

        # ------------------- normalize neck --------------------- #
        neck_new_cor, _ = length_norm(pose[0], pose[1], pose[3], pose[4],
                               get_distance(pose[0], pose[1], pose[3], pose[4]) * rig_ratio)
        # neck_new_cor, _ = length_norm(pose[0], pose[1], pose[3], pose[4],
        #                        neck_len_mean)
        pose[0] = neck_new_cor[0]
        pose[1] = neck_new_cor[1]
        
        # right arm
        rig_arm_new_cor, _ = length_norm(pose[9], pose[10], pose[6], pose[7],
                               get_distance(pose[9], pose[10], pose[6], pose[7]) * rig_ratio)
        rig_diff_x = rig_arm_new_cor[0] - pose[9]
        rig_diff_y = rig_arm_new_cor[1] - pose[10]
        
        pose[9] = rig_arm_new_cor[0]
        pose[10] = rig_arm_new_cor[1]
        # re-locate right hand coordination
        pose[12] += rig_diff_x
        pose[13] += rig_diff_y
        
        # right hand
        rig_hand_new_cor, _ = length_norm(pose[12], pose[13], pose[9], pose[10],
                                         get_distance(pose[12], pose[13], pose[9], pose[10]) * rig_ratio)
        pose[12] = rig_hand_new_cor[0]
        pose[13] = rig_hand_new_cor[1]
        
        # left arm
        lef_arm_new_cor, _ = length_norm(pose[18], pose[19], pose[15], pose[16],
                                         get_distance(pose[18], pose[19], pose[15], pose[16]) * lef_ratio)
        lef_diff_x = lef_arm_new_cor[0] - pose[18]
        lef_diff_y = lef_arm_new_cor[1] - pose[19]
        
        pose[18] = lef_arm_new_cor[0]
        pose[19] = lef_arm_new_cor[1]
        # re-locate left hand coordination
        pose[21] += lef_diff_x
        pose[22] += lef_diff_y
        
        # left hand
        lef_hand_new_cor, _ = length_norm(pose[21], pose[22], pose[18], pose[19],
                                         get_distance(pose[21], pose[22], pose[18], pose[19]) * lef_ratio)
        pose[21] = lef_hand_new_cor[0]
        pose[22] = lef_hand_new_cor[1]
        
    # save explanded pose pickle 
    print('[INFO] Save shoulder norm pose.')
    torch.save(normalized, './processed_data/sh_norm.pickle')
    # exit(-1)
    return normalized, length


def build_emb_table(emb_path, target_vocab):

    def load_emb_file(emb_path):
        vects = []
        idx = 0
        word2idx = dict()
        idx2word = dict()
        with open(emb_path, 'r') as f:
            for l in tqdm(f):
                line = l.split()
                word = line[0]
                w_vec = np.array(line[1:]).astype(np.float)

                vects.append(w_vec)
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
        
        return np.array(vects), word2idx, idx2word

    vects, word2idx, idx2word = load_emb_file(emb_path)
    dim = vects.shape[1]

    emb_tb = np.zeros((len(target_vocab), dim))
    for k, v in target_vocab.items():
        try:
            emb_tb[v] = vects[word2idx[k]]
        except KeyError:
            emb_tb[v] = np.random.normal(scale=0.6, size=(dim, )) 

    return emb_tb


def main():
    def get_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default="./data/ted_gesture_dataset_train.pickle")
    parser.add_argument('-valid_src', default="./data/ted_gesture_dataset_val.pickle")
    parser.add_argument('-pose_dir', default="./processed_data/neck_loc.pickle")
    parser.add_argument('-data_size', default=10000)
    parser.add_argument('-sample_rate', default=1)
    parser.add_argument('-save_data', default="./processed_data/preprocessing.pickle")
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-pca_components', type=int, default=10)
    parser.add_argument('-emb_src', default="./data/glove.6B.300d.txt")
    
    parser.add_argument('-mode', default='pca')

    opt = parser.parse_args()

    print('[INFO] {} selected.'.format(opt.mode))

    # display pca subspace or sample pos from dataset
    if opt.mode == 'pca':
        data = torch.load(opt.save_data)
        pca = data['pca']
        factor = 3
    
        m_0 = np.diag([factor]*10)
        m_1 = np.diag([factor / 2]*10)
        m_2 = np.diag([factor / 2 * -1]*10)
        m_3 = np.diag([factor * -1]*10)
        sample = np.concatenate((m_0, m_1, m_2, m_3), axis=0)
    
        trans_pos = pca.inverse_transform(sample)
        display_multi_poses(trans_pos)
        plt.show()
        exit(-1)
    
    elif opt.mode == 'display':
        poses = torch.load(opt.pose_dir)
        poses = [p for p in poses if (p[6] < 0.18)]
        p = poses[0]
        print("count: {}".format(len(poses)))
        print("sh_len:{}".format(get_distance(p[6], p[7], p[3], p[4])))
        # poses = random.sample(list(poses), k=30)
        if len(poses) < 30:
            poses = poses[:]
            display_multi_poses(np.array(poses), col=1)
        else:
            poses = poses[:30]
            display_multi_poses(np.array(poses), col=10)
        plt.show()
        exit(-1)

    # get train set
    train_data = loadpickle(opt.train_src, opt.data_size)
    val_data = loadpickle(opt.valid_src, opt.data_size)

    train_src_insts, train_tgt_insts = get_data(train_data, opt.sample_rate)
    valid_src_insts, valid_tgt_insts = get_data(val_data, opt.sample_rate)

    print('[INFO] normalize target pose instance')
    norm_tr_tgt, tr_l = tgt_insts_normalize(train_tgt_insts)
    norm_val_tgt, val_l = tgt_insts_normalize(valid_tgt_insts)

    print('[INFO] Convert target pose instance into normalized pca values')
    pca, train_tgt_insts = run_PCA_train_tgt(norm_tr_tgt, tr_l, opt.pca_components)
    valid_tgt_insts = run_PCA_val_tgt(pca, norm_val_tgt, val_l, opt.pca_components)

    print('[INFO] Build vocabulary.')
    word2idx = build_vocab_idx(train_src_insts, opt.min_word_count)

    print('[INFO] Build embedding table.')
    emb_tb = build_emb_table(opt.emb_src, word2idx)

    print('[INFO] Convert source word instance into seq for word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_insts, word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_insts, word2idx)

    data = {
        'settings': opt,
        'dict': word2idx,
        'emb_tbl': emb_tb,
        'pca': pca,
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts
        },
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts
        }
    }

    print('[INFO] Dumping the processed data to pickle file: {}'.format(opt.save_data))
    torch.save(data, opt.save_data)
    print('[INFO] Finish.')


def check_certain(src_inst, tgt_insts):
    pass


if __name__ == '__main__':
    main()