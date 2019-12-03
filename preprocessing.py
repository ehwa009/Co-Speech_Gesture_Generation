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



    def get_new_pos(length, fix_x, fix_y, var_y, sh):
        if sh == 'left':
            return math.sqrt( length**2 - (var_y - fix_y)**2 ) + fix_x
        else:
            return -1 * math.sqrt( length**2 - (var_y - fix_y)**2 ) + fix_x

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


def run_PCA_train_tgt(tgt_insts, n_components):
    

    
    tgt_insts, lengths = tgt_insts_normalize(tgt_insts)



    pca = PCA(n_components=n_components)
    pca_tgt = pca.fit_transform(tgt_insts)

 
    ori_tgt = []
    # initial index of expanded pose array
    start = 0
    for l in lengths:
        # create empty array to store pca poses
        pca_skel = np.zeros((1, n_components))
        sel_p = pca_tgt[start:start+l]
        # for i in range(sel_p.shape[0]):
        #     sel_p[i][2] = 0.00
        # stack
        ori_tgt.append(sel_p)
        # change index
        start = l
    
    return pca, ori_tgt


def run_PCA_val_tgt(pca, tgt_insts, n_components):
    tgt_insts, lengths = tgt_insts_normalize(tgt_insts)
    pca_tgt = pca.transform(tgt_insts)

    ori_tgt = []
    # initial index of expanded pose array
    start = 0
    for l in lengths:
        # create empty array to store pca poses
        pca_skel = np.zeros((1, n_components))
        sel_p = pca_tgt[start:start+l]
        # for i in range(sel_p.shape[0]):
        #     sel_p[i][2] = 0.00
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
        return dist * np.array([math.cos(theta), math.sin(theta)]) + np.array([point[0], point[1]])

    def length_norm(var_x, var_y, fix_x, fix_y, mean_len):
        angle = get_theta(var_x, var_y, fix_x, fix_y)
        new_cor = get_new_cor(angle,
                    get_distance(var_x, var_y, fix_x, fix_y),
                    [var_x, var_y])

        return new_cor


    tmp = []
    length = []
    # expand poses list
    for pose in tgt_insts:
        length.append(len(pose))
        for p in pose:
            tmp.append(p)
    
    # convert list to np array
    tmp = np.array(tmp)
    # normalized with specific scale
    normalized = preprocessing.normalize(tmp, norm='l2') * 50

    # relocate x and y of neck coordinate
    mean_val_pose = np.mean(normalized, axis=0)

    # get mean dist of each shoulders
    rig_sh_len_mean = get_distance(((mean_val_pose[3], mean_val_pose[4]),
                                    (mean_val_pose[6], mean_val_pose[7])))
    lef_sh_len_mean = get_distance(((mean_val_pose[3], mean_val_pose[4]),
                                    (mean_val_pose[15], mean_val_pose[16])))
    neck_len_mean = get_distance(((mean_val_pose[3], mean_val_pose[4]),
                                    (mean_val_pose[0], mean_val_pose[1])))
    rig_arm_len_mean = get_distance(((mean_val_pose[3], mean_val_pose[4]),
                                    (mean_val_pose[6], mean_val_pose[7])))
    rig_hand_len_mean = get_distance(((mean_val_pose[9], mean_val_pose[10]),
                                    (mean_val_pose[12], mean_val_pose[13])))
    lef_arm_len_mean = get_distance(((mean_val_pose[3], mean_val_pose[4]),
                                    (mean_val_pose[15], mean_val_pose[16])))
    lef_hand_len_mean = get_distance(((mean_val_pose[18], mean_val_pose[19]),
                                    (mean_val_pose[21], mean_val_pose[22])))

    for pose in normalized:
        # ------------------- re-coordinate neck --------------------- #
        neck_diff_x = mean_val_pose[3] - pose[3]
        neck_diff_y = mean_val_pose[4] - pose[4]
        for i in range(len(pose)):
            if i % 3 == 0: # x
                pose[i] += neck_diff_x
            elif i % 3 == 1: # y
                pose[i] += neck_diff_y
        # modify neck x and y pos
        pose[3] = mean_val_pose[3]
        pose[4] = mean_val_pose[4]
        # # ------------------- normalize shoulder --------------------- #
        # # get theta
        # rig_angle = get_theta( ((pose[6], pose[7]),
        #                         (pose[3], pose[4])) )
        # lef_angle = get_theta( ((pose[15], pose[16]),
        #                         (pose[3], pose[4])) )
        #
        # rig_len = get_distance( ((pose[3], pose[4]), (pose[6], pose[7])) )
        # lef_len = get_distance( ((pose[3], pose[4]), (pose[15], pose[16])) )
        # rig_ratio = rig_len / rig_sh_len_mean
        # lef_ratio = lef_len / lef_sh_len_mean
        # print('TEST')
        #
        # new_rig_sh_pos = get_new_cor(rig_angle, rig_sh_len_mean, pose[6:8])
        # new_lef_sh_pos = get_new_cor(lef_angle, lef_sh_len_mean, pose[15:17])
        #
        # pose[6] = new_rig_sh_pos[0]  # x
        # pose[7] = new_rig_sh_pos[1]  # y
        # pose[15] = new_lef_sh_pos[0]  # x
        # pose[16] = new_lef_sh_pos[1]  # y
        #
        # # neck length
        # neck_len = get_distance( ((pose[3], pose[4]), (pose[0], pose[1])) ) + neck_len_mean * (1 - rig_ratio)
        # angle = get_theta( ((pose[0], pose[1]),
        #                     (pose[3], pose[4])) )
        # new_neck_cor = get_new_cor(angle, neck_len, pose[0:2])
        # pose[0] = new_neck_cor[0]
        # pose[1] = new_neck_cor[1]
        #
        # # right arm
        # arm_len = get_distance( ((pose[9], pose[10]), (pose[6], pose[7])) ) + rig_arm_len_mean * (1 - rig_ratio)
        # angle = get_theta( ((pose[9], pose[10]),
        #                     (pose[6], pose[7])) )
        # new_rig_arm_cor = get_new_cor(angle, arm_len, pose[9:11])
        # pose[9] = new_rig_arm_cor[0]
        # pose[10] = new_rig_arm_cor[1]
        #
        # # right hand
        # hand_len = get_distance(((pose[9], pose[10]), (pose[12], pose[13]))) + rig_hand_len_mean * (1 - rig_ratio)
        # angle = get_theta(((pose[12], pose[13]),
        #                    (pose[9], pose[10])))
        # new_rig_hand_cor = get_new_cor(angle, hand_len, pose[12:14])
        # pose[12] = new_rig_hand_cor[0]
        # pose[13] = new_rig_hand_cor[1]
        #
        # # left arm
        # arm_len = get_distance(((pose[15], pose[16]), (pose[18], pose[19]))) + lef_arm_len_mean * (1 - lef_ratio)
        # angle = get_theta(((pose[18], pose[19]),
        #                    (pose[15], pose[16])))
        # new_lef_arm_cor = get_new_cor(angle, arm_len, pose[18:20])
        # pose[18] = new_lef_arm_cor[0]
        # pose[19] = new_lef_arm_cor[1]
        #
        # # left hand
        # hand_len = get_distance(((pose[18], pose[19]), (pose[21], pose[22]))) + lef_hand_len_mean * (1 - lef_ratio)
        # angle = get_theta(((pose[21], pose[22]),
        #                    (pose[18], pose[19])))
        # new_rig_hand_cor = get_new_cor(angle, hand_len, pose[21:23])
        # pose[21] = new_rig_hand_cor[0]
        # pose[22] = new_rig_hand_cor[1]

        # display_pose(pose)
        # plt.show()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default="./data/ted_gesture_dataset_train.pickle")
    parser.add_argument('-valid_src', default="./data/ted_gesture_dataset_val.pickle")
    parser.add_argument('-data_size', default=10000)
    parser.add_argument('-sample_rate', default=3)
    parser.add_argument('-save_data', default="./processed_data/preprocessing.pickle")
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-pca_components', type=int, default=10)
    parser.add_argument('-emb_src', default="./data/glove.6B.300d.txt")
    
    parser.add_argument('-display', type=bool, default=True)
    parser.add_argument('-display_pca', type=bool, default=False)

    opt = parser.parse_args()

    # display pca subspace or sample pos from dataset
    if opt.display:
        display_sample(opt)
        exit(-1)

    # get train set
    train_data = loadpickle(opt.train_src, opt.data_size)
    val_data = loadpickle(opt.valid_src, opt.data_size)

    train_src_insts, train_tgt_insts = get_data(train_data, opt.sample_rate)
    valid_src_insts, valid_tgt_insts = get_data(val_data, opt.sample_rate)

    print('[INFO] Build vocabulary.')
    word2idx = build_vocab_idx(train_src_insts, opt.min_word_count)

    print('[INFO] Build embedding table.')
    emb_tb = build_emb_table(opt.emb_src, word2idx)

    print('[INFO] Convert source word instance into seq for word index')
    train_src_insts = convert_instance_to_idx_seq(train_src_insts, word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_insts, word2idx)

    print('[INFO] Convert target pose instance into normalized pca values')
    pca, train_tgt_insts = run_PCA_train_tgt(train_tgt_insts, opt.pca_components)
    valid_tgt_insts = run_PCA_val_tgt(pca, valid_tgt_insts, opt.pca_components)

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


def display_sample(opt):
    data = torch.load(opt.save_data)
    pca = data['pca']
    if opt.display_pca:
        factor = 1.5
        m_0 = np.diag([factor]*10)
        m_1 = np.diag([factor / 2]*10)
        m_2 = np.diag([factor / 2 * -1]*10)
        m_3 = np.diag([factor * -1]*10)
        sample = np.concatenate((m_0, m_1, m_2, m_3), axis=0)
    else:
        sample = data['train']['tgt'][0][:30]
    
    trans_pos = pca.inverse_transform(sample)
    display_multi_poses(trans_pos)
    plt.show()


if __name__ == '__main__':
    main()