import numpy as np
import torch
from torch.autograd import Variable
from transformer import make_model, LabelSmoothing, NoamOpt, Batch, SimpleLossCompute, run_epoch, greedy_decode
use_gpu = torch.cuda.is_available()

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

def Synthetic_Data():
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    print(model)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if use_gpu:
        model.cuda()
        criterion.cuda()

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt), use_gpu)
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None), use_gpu))

    model.eval()
    src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
    src_mask = Variable(torch.ones(1, 1, 10) )
    if use_gpu:
        src = src.cuda()
        src_mask = src_mask.cuda()
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

if __name__ == '__main__':
    Synthetic_Data()