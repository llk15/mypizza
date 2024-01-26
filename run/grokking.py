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

DEVICE = 'cuda:0'
C = cfg.n_vocab
cfg.project = 'grokking_mlp_image'

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_SILENT"] = "false"
cfg.tqdm_disable = False

# dataset
full_dataset = MyDataset(C)
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
    run = wandb.init(reinit=True, project=cfg.project, group = str(cfg.dataset_seed), name=cfg.name, config=vars(cfg))
    run.watch(model, optimizer, log='all')
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

    # logits
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                total = total_loss = total_correct = 0
                for inputs, labels in full_loader:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(inputs)
                    if len(outputs.shape) == 3: # the last token for transformer
                        outputs = outputs[:,-1,:]
                    corrects = (outputs.argmax(dim=1)==labels).reshape([C,C]).cpu().numpy()
                    img = wandb.Image(corrects, mode='L', caption=f"corrects at epoch {epoch+1}")
            run.log({'corrects': img }, step=epoch+1)

    run.finish()
    return

for num_try in range(1):
    # random setting
    cfg.dataset_seed = random.randint(1000,9999)
    for i in range(1):
        cfg.model_seed = random.randint(1000,9999)
        for cfg.model_type in ['A', 'T']:
            cfg.name = '_'.join([cfg.model_type, str(cfg.model_seed)])
            print(cfg.name)

            group_path = PATH + str(cfg.dataset_seed) + '/'
            if not os.path.exists(group_path):
                os.mkdir(group_path)
            train(cfg)