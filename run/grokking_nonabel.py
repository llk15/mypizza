import torch
import random

import os

import wandb
from tqdm import tqdm

from myutils import *
from models import *
from configs import cfg

PATH = '/home/llk/pizza/save/grokking/'
if not os.path.exists(PATH):
    os.makedirs(PATH)

DEVICE = 'cuda:7'
C = cfg.n_vocab
cfg.project_base = 'grokking_nonabel'
cfg.epoch = 10000

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_SILENT"] = "false"
cfg.tqdm_disable = True

# dataset
full_dataset = MyDataset(cfg)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=C*C, shuffle=False)
train_size = int(cfg.frac * len(full_dataset))
test_size = len(full_dataset) - train_size

# train
def train(cfg):
# dataset
    torch.manual_seed(cfg.dataset_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=C*C, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=C*C, shuffle=False)
# model
    models={'A':MyModelA,'B':MyModelB,'C':MyModelC,'D':MyModelD,'X':MyModelX,'T':Transformer}
    torch.manual_seed(cfg.model_seed)
    model = models[cfg.model_type](cfg)
    model.to(DEVICE)
# optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = cross_entropy_high_precision
# run
    run = wandb.init(reinit=True, project=cfg.project, group=cfg.group, name=cfg.name, config=vars(cfg))
    # run.watch(model, optimizer, log='all')
    bar = tqdm(range(cfg.epoch), bar_format='{l_bar}{bar:30}{r_bar}', mininterval=1.0, disable=cfg.tqdm_disable)
    for epoch in bar:
        # train
        model.train()
        total = total_loss = total_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(outputs.shape) == 3: # the last token for transformer
                outputs = outputs[:,-1,:]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1)==labels).sum().item()
        train_loss = total_loss/total
        train_acc = total_correct/total
        log = ('epoch %d [TRAIN]loss: %.3g, acc: %.3f' % (epoch + 1, train_loss, train_acc))
        # test
        with torch.no_grad():
            model.eval()
            total = total_loss = total_correct = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                if len(outputs.shape) == 3: # the last token for transformer
                    outputs = outputs[:,-1,:]
                loss = criterion(outputs, labels)

                total += labels.size(0)
                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=1)==labels).sum().item()
            test_loss = total_loss/total
            test_acc = total_correct/total
            log += (' [TEST]]loss: %.3g, acc: %.3f' % (test_loss, test_acc))
        bar.set_description(log, refresh=False)
        # wandb
        run.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
        }, step=epoch+1)

    run.finish()
    return

for dataset_try in range(1):
    # random setting
    # cfg.dataset_seed = random.randint(1000,9999)
    cfg.dataset_seed = 4356
    cfg.project = cfg.project_base +'_'+ str(cfg.dataset_seed)
    cfg.model_type ='T'
    for cfg.frac in [0.7,0.8]:
        for cfg.weight_decay in [1.0,0.5,0.2,0.1]:
            for model_try in range(10):
                cfg.model_seed = random.randint(1000,9999)
                cfg.group = '_'.join([str(cfg.frac), str(cfg.weight_decay)])
                cfg.name = str(cfg.model_seed)
                print(cfg.group, cfg.name)

                train(cfg)