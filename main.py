import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import AverageMeter, seq2sen, val_check, save, load
from model import Transformer
from tqdm import tqdm
import logging
import time

def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True
    
    if not os.path.isdir('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.isdir('./results'):
        os.mkdir('./results')    
    if not os.path.isdir(os.path.join('./ckpt', args.name)):
        os.mkdir(os.path.join('./ckpt', args.name))
    if not os.path.isdir(os.path.join('./results', args.name)):
        os.mkdir(os.path.join('./results', args.name))
    if not os.path.isdir(os.path.join('./results', args.name, "log")):
        os.mkdir(os.path.join('./results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler("results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time()))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
        
    args.logger.info("[{}] starts".format(args.name))
    
    # 1. load data
    
    args.logger.info("loading data...")
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # 2. setup
    
    args.logger.info("setting up...")

    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # transformer config
    d_e = 512      # embedding size
    d_q = 64       # query size (= key, value size)
    d_h = 2048        # hidden layer size in feed forward network
    num_heads = 8
    num_layers = 6    # number of encoder/decoder layers in encoder/decoder
    
    args.sos_idx = sos_idx
    args.eos_idx = eos_idx
    args.pad_idx = pad_idx
    args.max_length = max_length
    args.src_vocab_size = src_vocab_size
    args.tgt_vocab_size = tgt_vocab_size
    args.d_e = d_e
    args.d_q = d_q
    args.d_h = d_h
    args.num_heads = num_heads
    args.num_layers = num_layers
    
    model = Transformer(args)
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    if args.load:
        model.load_state_dict(load(args, args.ckpt))
        
    # 3. train / test
    
    if not args.test:
        # train
        args.logger.info("starting training")
        acc_val_meter = AverageMeter(name="Acc-Val (%)", save_all=True, save_dir=os.path.join('results', args.name))
        train_loss_meter = AverageMeter(name="Loss", save_all=True, save_dir=os.path.join('results', args.name))
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        for epoch in range(1, 1 + args.epochs):
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter()
            for src_batch, tgt_batch in tqdm(train_loader):
                # src_batch: (batch x source_length), tgt_batch: (batch x target_length)
                optimizer.zero_grad()
                src_batch, tgt_batch = torch.LongTensor(src_batch).to(args.device), torch.LongTensor(tgt_batch).to(args.device)
                batch = src_batch.shape[0]
                # split target batch into input and output
                tgt_batch_i = tgt_batch[:,:-1]
                tgt_batch_o = tgt_batch[:,1:]
                
                pred = model(src_batch.to(args.device), tgt_batch_i.to(args.device))
                loss = loss_fn(pred.contiguous().view(-1, tgt_vocab_size), tgt_batch_o.contiguous().view(-1))
                loss.backward()
                optimizer.step()
                
                train_loss_tmp_meter.update(loss / batch, weight=batch)

            train_loss_meter.update(train_loss_tmp_meter.avg)
            spent_time = time.time() - spent_time
            args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
            
            # validation
            model.eval()
            acc_val_tmp_meter = AverageMeter()
            spent_time = time.time()

            for src_batch, tgt_batch in tqdm(valid_loader):
                src_batch, tgt_batch = torch.LongTensor(src_batch), torch.LongTensor(tgt_batch)
                tgt_batch_i = tgt_batch[:, :-1]
                tgt_batch_o = tgt_batch[:, 1:]
      
                with torch.no_grad():
                    pred = model(src_batch.to(args.device), tgt_batch_i.to(args.device))

                corrects, total = val_check(pred.max(dim=-1)[1].cpu(), tgt_batch_o)
                acc_val_tmp_meter.update(100 * corrects / total, total)
            
            spent_time = time.time() - spent_time
            args.logger.info("[{}] validation accuracy: {:.1f} %, took {} seconds".format(epoch, acc_val_tmp_meter.avg, spent_time))
            acc_val_meter.update(acc_val_tmp_meter.avg)
            
            if epoch % args.save_period == 0:
                save(args, "epoch_{}".format(epoch), model.state_dict())
                acc_val_meter.save()
                train_loss_meter.save()
    else:
        # test
        args.logger.info("starting test")
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)
        pred_list = []
        model.eval()
        
        for src_batch, tgt_batch in test_loader:
            #src_batch: (batch x source_length)
            src_batch = torch.Tensor(src_batch).long().to(args.device)
            batch = src_batch.shape[0]           
            pred_batch = torch.zeros(batch, 1).long().to(args.device)
            pred_mask = torch.zeros(batch, 1).bool().to(args.device)    # mask whether each sentece ended up
            
            with torch.no_grad():
                for _ in range(args.max_length):
                    pred = model(src_batch, pred_batch)   # (batch x length x tgt_vocab_size)
                    pred[:, :, pad_idx] = -1              # ignore <pad>
                    pred = pred.max(dim=-1)[1][:, -1].unsqueeze(-1)   # next word prediction: (batch x 1)
                    pred = pred.masked_fill(pred_mask, 2).long()      # fill out <pad> for ended sentences
                    pred_mask = torch.gt(pred.eq(1) + pred.eq(2), 0)
                    pred_batch = torch.cat([pred_batch, pred], dim=1)
                    if torch.prod(pred_mask) == 1:
                        break
                        
            pred_batch = torch.cat([pred_batch, torch.ones(batch, 1).long().to(args.device) + pred_mask.long()], dim=1)   # close all sentences
            pred_list += seq2sen(pred_batch.cpu().numpy().tolist(), tgt_vocab)
            
        with open('results/pred.txt', 'w', encoding='utf-8') as f:
            for line in pred_list:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')


if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=5)
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='_')
    parser.add_argument(
        '--load',
        action='store_true')
    
    args = parser.parse_args()

        
    main(args)
