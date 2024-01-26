import torch
import torch.nn as nn
import torch.nn.functional as F

##### dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.n_vocab = cfg.n_vocab
        self.abel = cfg.abel
        self.data = []
        M = self.n_vocab
        for i in range(self.n_vocab):
            if self.abel:
                M = i
            for j in range(M):
                self.data.append([i,j])
    def __getitem__(self, index):
        return torch.tensor(self.data[index],dtype=int),sum(self.data[index]) % self.n_vocab
    def __len__(self):
        return len(self.data)

    
##### functions
def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    return -prediction_logprobs.mean()


def tatol_norm(model):
    su=0
    for t in model.parameters():
        su+=(t*t).sum().item()
    return su**0.5

def norms(model):
    tmp = {}
    for name, para in model.named_parameters():
        tmp[name] = (para*para).sum().item()**0.5
    return tmp