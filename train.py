import argparse
import torch
import time
import torch.nn.functional as F
import os

from torch import optim
from tqdm import tqdm
from dataset import TedDataset, collate_fn
from functools import partial
from transformer.models import Transformer
from seq2pose.models import Seq2Pose

# to prevent error from num_workers option
torch.multiprocessing.set_sharing_strategy('file_system')

def cust_loss(output, target, alpha, beta):
    n_element = output.numel()
    
    # mse
    mse_loss = F.mse_loss(output, target)
    
    # countinous motion
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss /= 100

    # motion variance
    norm = torch.norm(output, 2, 1)
    var_loss = -torch.sum(norm) / n_element
    var_loss /= 1

    # final loss
    loss = mse_loss + alpha * cont_loss + beta * var_loss

    return loss


def train(model, training_data, validation_data, optim, device, opt, start_i=0):
    ''' Start traning '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '{}_train.log'.format(opt.model)
        log_valid_file = opt.log + '{}_valid.log'.format(opt.model)
        print('[INFO] Training performance will be written to file: {} and {}'.format(
                                                            log_train_file, log_valid_file))

        if not(os.path.exists(log_train_file) and os.path.exists(log_valid_file)):
            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss\n')
                log_vf.write('epoch,loss\n')

    valid_loss_list = []
    for epoch_i in range(start_i, opt.epoch):
        print('[ Epoch: {} ]'.format(epoch_i))

        start = time.time()
        train_loss = train_epoch(model, training_data, optim, device, opt)
        print('\t- (Training)   loss: {loss: 8.5f}, elapse: {elapse:3.3f}'.format(
                                    loss=train_loss, elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss = eval_epoch(model, validation_data, device, opt)
        print('\t- (Validation)   loss: {loss: 8.5f}, elapse: {elapse:3.3f}'.format(
                                    loss=valid_loss, elapse=(time.time()-start)/60))

        valid_loss_list += [valid_loss]

        # define parameter to save trained model
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_tr_loss_{epoch}_{train_loss: 3.3f}.chkpt'.format(
                                                                                epoch=epoch_i,
                                                                                train_loss=train_loss)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_loss >= max(valid_loss_list):
                    torch.save(checkpoint, model_name)
                    print('\t[INFO] The checkpoint file has been updated.')
            elif opt.save_mode == 'interval':
                if (epoch_i % opt.save_interval) == 0: 
                    model_name = opt.save_model + '_tr_loss_{epoch}_{train_loss: 3.3f}.chkpt'.format(
                                                                                    epoch=epoch_i,
                                                                                    train_loss=train_loss)
                    torch.save(checkpoint, model_name)
                    print('\t[INFO] The checkpoint file has been saved.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss))
                log_vf.write('{epoch},{loss: 8.5f}\n'.format(
                    epoch=epoch_i, loss=valid_loss))


def eval_epoch(model, validation_data, device, opt):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=' - (Validation)', leave=False):
            batch_loss = 0
            n_motion = 0
            for src_seq, src_len, src_pos, tgt_seq, tgt_pos in batch:
                src_seq = src_seq.to(device)
                tgt_seq = tgt_seq.to(device)
                # predict
                if opt.model == "transformer": # todo
                    pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                elif opt.model == 'seq2pos':
                    pred, ans = model(opt, src_seq, src_len, tgt_seq, device)
                    loss = cust_loss(pred, ans, opt.alpha, opt.beta)
                    # note keeping
                    batch_loss += loss.item()
                    n_motion += 1
            total_loss += batch_loss/n_motion
        
        return total_loss


def train_epoch(model, training_data, optim, device, opt):
    model.train()

    total_loss = 0
    for batch in tqdm(training_data, mininterval=2, desc=' - (Training)', leave=False):
        batch_loss = 0
        n_motion = 0
        for src_seq, src_len, src_pos, tgt_seq, tgt_pos in batch:
            # make gradient zero
            optim.zero_grad()
            # processed dataset
            src_seq = src_seq.to(device)
            tgt_seq = tgt_seq.to(device)
            src_pos = src_pos.to(device)
            tgt_pos = tgt_pos.to(device)
            # predict
            if opt.model == "transformer": # todo
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            elif opt.model == 'seq2pos':
                pred, ans = model(opt, src_seq, src_len, tgt_seq, device)
                loss = cust_loss(pred, ans, opt.alpha, opt.beta)
                loss.backward()

            # optimize
            optim.step()
            # note keeping
            batch_loss += loss.item()
            n_motion += 1

        total_loss += batch_loss/n_motion
    
    return total_loss


def main():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('-data', default='./processed_data/preprocessing.pickle')
    parser.add_argument('-epoch', type=int, default=530)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-n_workers', type=int, default=0)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-model', default='seq2pos')
    parser.add_argument('-save_model', default='./trained_model/seq2pos')
    parser.add_argument('-save_mode', default='interval')
    parser.add_argument('-save_interval', type=int, default=10)
    parser.add_argument('-log', default='./log/')
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-chkpt', default=False)
    
    # seq2pos args
    parser.add_argument('-alpha', type=int, default=0.1)
    parser.add_argument('-beta', type=int, default=1)
    parser.add_argument('-hidden_size', type=int, default=200)
    parser.add_argument('-bidirectional', type=bool, default=True)
    parser.add_argument('-tf_ratio', type=int, default=0.4)
    parser.add_argument('-n_enc_layers', type=int, default=2)
    parser.add_argument('-n_dec_layers', type=int, default=1)
    parser.add_argument('-pre_motions', type=int, default=10)
    parser.add_argument('-estimation_motions', type=int, default=20)
    parser.add_argument('-frame_duration', type=int, default=1/12)
    parser.add_argument('-speech_sp', type=int, default=2.5) # assume speech speed is 2.5 wps

    # transformer args
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-d_enc_model', type=int, default=300)
    parser.add_argument('-d_dec_model', type=int, default=10)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    
    opt = parser.parse_args()

    ############################################
    #             Loading Dataset              #
    ############################################
    data = torch.load(opt.data)

    training_data, validation_data = prepare_dataloaders(data, opt)
    opt.scr_vocab_size = training_data.dataset.scr_vocab_size
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.chkpt:
        print('[INFO] continue train from checkpoint from:{}'.format(opt.chkpt))
        model_info = torch.load(opt.chkpt)
        state = model_info['model']
        opt = model_info['settings']
        start_i = model_info['epoch']

        ############################################
        #               Prepare Model              #
        ############################################
        if opt.model == 'transformer':
            print('[INFO] transformer model selected.')
            model = Transformer(
                emb_matrix=data['emb_tbl'],
                n_src_vocab=opt.scr_vocab_size,
                len_max_seq=8, # todo
                d_enc_model=opt.d_enc_model,
                d_dec_model=opt.d_dec_model,
                d_inner=opt.d_inner_hid,
                n_layers=opt.n_layers,
                d_k=opt.d_k,
                d_v=opt.d_v,
                dropout=opt.dropout).to(device)
        elif opt.model == 'seq2pos':
            print('[INFO] seq2pos model selected.')
            model = Seq2Pose(
                word_emb=data['emb_tbl'],
                batch_size=opt.batch_size,
                hidden_size=opt.hidden_size,
                n_enc_layers=opt.n_enc_layers,
                n_dec_layers=opt.n_dec_layers,
                bidirectional=opt.bidirectional,
                dropout=opt.dropout,
                out_dim=data['pca'].n_components).to(device)
        else:
            print("[ERROR] undefined model.")
        print('[INFO] load state dict')
        model.load_state_dict(state)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        train(model, training_data, validation_data, optimizer, device, opt, start_i=start_i+1)

    else:
        ############################################
        #               Prepare Model              #
        ############################################
        if opt.model == 'transformer':
            print('[INFO] transformer model selected.')
            model = Transformer(
                emb_matrix=data['emb_tbl'],
                n_src_vocab=opt.scr_vocab_size,
                len_max_seq=8, # todo
                d_enc_model=opt.d_enc_model,
                d_dec_model=opt.d_dec_model,
                d_inner=opt.d_inner_hid,
                n_layers=opt.n_layers,
                d_k=opt.d_k,
                d_v=opt.d_v,
                dropout=opt.dropout).to(device)
        elif opt.model == 'seq2pos':
            print('[INFO] seq2pos model selected.')
            model = Seq2Pose(
                word_emb=data['emb_tbl'],
                batch_size=opt.batch_size,
                hidden_size=opt.hidden_size,
                n_enc_layers=opt.n_enc_layers,
                n_dec_layers=opt.n_dec_layers,
                bidirectional=opt.bidirectional,
                dropout=opt.dropout,
                out_dim=data['pca'].n_components).to(device)
        else:
            print("[ERROR] undefined model.")

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        train(model, training_data, validation_data, optimizer, device, opt)

    ############################################
    #            Prepare Dataloader            #
    ############################################
def prepare_dataloaders(data, opt):
    train_loader = torch.utils.data.DataLoader(
        TedDataset(
            src_word2idx=data['dict'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']
            ),
            num_workers=opt.n_workers,
            batch_size=opt.batch_size,
            collate_fn=partial(collate_fn, opt=opt),
            shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TedDataset(
            src_word2idx=data['dict'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']
            ),
            num_workers=opt.n_workers,
            batch_size=opt.batch_size,
            collate_fn=partial(collate_fn, opt=opt))

    return train_loader, valid_loader


if __name__ == '__main__':
    main()