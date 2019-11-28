import argparse
import torch
import pickle
import math
import transformer.constant as Constants
import numpy as np
from tqdm import tqdm

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

    def get_distance(points):
        p1 = points[0]
        p2 = points[1]

        return math.sqrt( ((p1[0] - p2[0]) ** 2) + ((p1[1]-p2[1]) ** 2) )

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

        sh_pos_y = 330.72
        sh_len = 154
        
        for p in poses:
            # remove if there is 0 value in poses
            if not(0 in p):
                # make shoulder has same value
                diff_neck = p[4] - sh_pos_y
                diff_sh_1 = p[16] - sh_pos_y
                diff_sh_2 = p[7] - sh_pos_y

                p[4] = sh_pos_y
                p[7] = sh_pos_y
                p[16] = sh_pos_y

                p[1] = p[1] - diff_neck
                p[10] = p[10] - diff_sh_2
                p[13] = p[13] - diff_sh_2
                p[19] = p[19] - diff_sh_1
                p[22] = p[22] - diff_sh_1


                shoulder1 = ( (p[3], p[4]), (p[15], p[16]) )
                shoulder2 = ( (p[3], p[4]), (p[6], p[7]) )
                
                dist1 = get_distance(shoulder1)
                dist2 = get_distance(shoulder2)

                # get new left and right factor
                left_factor = get_new_pos(sh_len, p[3], p[4], p[16], 'left')
                right_factor = get_new_pos(sh_len, p[3], p[4], p[7], 'right')

                diff_x_left = p[15] - left_factor
                diff_x_right = p[6] - right_factor

                p[15] = left_factor
                p[6] = right_factor

                p[9] -= diff_x_right
                p[12] -= diff_x_right

                p[18] -= diff_x_left
                p[21] -= diff_x_left

                shoulder1_af = ( (p[3], p[4]), (p[15], p[16]) )
                shoulder2_af = ( (p[3], p[4]), (p[6], p[7]) )

                # dist1_af = self.get_distance(shoulder1_af)
                # dist2_af = self.get_distance(shoulder2_af)

                dist1_list.append(dist1)
                dist2_list.append(dist2)

                
                # if (p[1] < p[4]) and ((p[4]-p[1])>100) and ((p[4]-p[1])<200):
                if (p[1] < p[4]):
                    tmp_poses.append(p)  
                
        # sampling 10fps
        tmp_poses = tmp_poses[::sampling_rate] 
            
        # selecte dataset with below condition;
        #                   1. pose seq must be longer than word seq
        #                   2. word seq has more than 12 (6*2)
        if (2 * len(sentence) < len(tmp_poses)) and (len(sentence) > 6*2):
            x_train.append(sentence)
            y_train.append(tmp_poses)

    dist1_list = np.array(dist1_list)
    dist2_list = np.array(dist2_list)
    print('[INFO] dataset desc.')
    print("\tparis: {}".format(len(x_train)))
    
    print("\tmax seq in x: {}".format(len(max(x_train, key=len))))
    print("\tmin seq in x: {}".format(len(min(x_train, key=len))))

    print("\tmax seq in y: {}".format(len(max(y_train, key=len))))
    print("\tmin seq in y: {}\n".format(len(min(y_train, key=len))))

    # print("dist1 mean: {}, dist2 mean: {}".format(np.mean(dist1_list), np.mean(dist2_list)))

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

    opt = parser.parse_args()

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


if __name__ == '__main__':
    main()