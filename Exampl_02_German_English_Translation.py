import numpy as np
import torch
from torch.autograd import Variable
from transformer import *
import torch.nn as nn
import spacy
from torchtext import data, datasets
use_gpu = torch.cuda.is_available()
#use_gpu = False
use_multi_gpu = False

#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
# or
# pip install de_core_news_sm-2.2.5.tar.gz
# pip install en_core_web_sm-2.2.5.tar.gz
# model: https://s3.amazonaws.com/opennmt-models/iwslt.pt
# For data loading.

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

def get_dataset(batch_size = 12000):
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    train_iter = MyIterator(train,
                            batch_size=batch_size,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=True)
    valid_iter = MyIterator(val,
                            batch_size=batch_size,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=False)

    return train_iter, valid_iter, TGT, SRC


def train(train_iter, valid_iter, TGT, SRC):
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

    if use_gpu:
        model.cuda()
        criterion.cuda()

        if use_multi_gpu:  # GPUs to use
            devices = [0]
            model_par = nn.DataParallel(model, device_ids=devices)

            model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            for epoch in range(10):
                model_par.train()
                run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                          MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))

                model_par.eval()
                loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                                 MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
                print(loss)

    if use_multi_gpu is False:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                      SimpleLossCompute(model.generator, criterion, model_opt), use_gpu)
            model.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                             SimpleLossCompute(model.generator, criterion, None), use_gpu)
            print(loss)

def test(valid_iter, SRC, TGT):
    model = torch.load("iwslt.pt")
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)

        if use_gpu:
            src = src.cuda()
            src_mask = src_mask.cuda()
        out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"])

        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break

def German_English_Translation():
    train_iter, valid_iter, TGT, SRC = get_dataset(batch_size = 1200)
    train(train_iter, valid_iter, TGT, SRC)
    test(valid_iter, SRC, TGT)

if __name__ == '__main__':
    German_English_Translation()