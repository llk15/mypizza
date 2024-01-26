# adapted from https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20 (https://arxiv.org/abs/2301.05217), thanks!
import torch
import torch.optim as optim
import json
import numpy as np
from tqdm import tqdm
import random

from functools import *
import wandb

from myutils import *
from models import *
from configs import config

PATH = '/home/llk/pizza/save/'
DEVICE = 'cuda:3'
C = config.n_vocab

# silence the terminal output
import os
os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_SILENT"] = "false"
config.tqdm_disable = True

# dataset
full_dataset = MyDataset(C)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=C*C, shuffle=False)
train_size = int(config.frac * len(full_dataset))
test_size = len(full_dataset) - train_size


def train(config):
    # dataset
    torch.manual_seed(config.dataset_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=C*C, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=C*C, shuffle=False)
    # model
    torch.manual_seed(config.model_seed)
    model = (Linearformer if config.use_linear else Transformer)(
        num_layers=config.n_layers,
        num_heads=config.n_heads,
        d_model=config.d_model,
        d_head=config.d_model//config.n_heads,
        attn_coeff=1.,  # weighted attention mask
        n_vocab=config.n_vocab,
        act_type=config.act_fn,
        n_ctx=2, # contex length
    )
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(),lr=config.lr, weight_decay=config.weight_decay, betas=(0.9,0.98))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1)) # 10 epoch warmup
    criterion = cross_entropy_high_precision

    # run
    run = wandb.init(reinit=True, name=config.run_name, group=str(config.dataset_seed) , config=vars(config), project='transformer_head')
    bar = tqdm(range(config.epoch), bar_format='{l_bar}{bar:30}{r_bar}', mininterval=1.0, disable=config.tqdm_disable)
    for epoch in bar:
        for inputs, labels in train_loader:            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)[:,-1,:]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_acc = (outputs.argmax(dim=1)==labels).float().mean().item()
            log = ('epoch %d [TRAIN]loss: %.3g, acc: %.3f' % (epoch + 1, loss.item(), train_acc))

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)[:,-1,:]
                test_loss = criterion(outputs, labels)
                test_acc = (outputs.argmax(dim=1)==labels).float().mean().item()

        cur_norm = norm(model)    
        log += (' [TEST]loss: %.3g acc: %.3f [MODEL] norm: %.3f' % (test_loss.item(), test_acc, cur_norm))
        
        
        bar.set_description(log, refresh=False)
        run.log({'train_loss': loss.item(),
                 'test_loss': test_loss.item(),
                 'test_acc':test_acc,
                 'norm': cur_norm,})
    # evaluate
    y_logits = np.zeros([C,C]) # logits for target logits
    ab_logits = np.zeros([C,C]) # output(A+B)_output(A-B) matrix for target logits
    for x,y in full_loader:
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        with torch.inference_mode():
            model.eval()
            logits = model(x)[:,-1,:] # [num_sample, C]
            target_logits = logits[list(range(len(x))),y].cpu() # [num_sample]
            x = x.cpu()
    for logit,sample in zip(target_logits, x):
        A,B = sample.tolist()
        y_logits[A][B] = logit.item()
        ab_logits[(A+B)%C][(A-B)%C] = logit.item()
    distance_dependency = ab_logits.std(axis=0).mean()/ab_logits.std()

    # save
    run_name = config.run_name
    torch.save(model.state_dict(), PATH + f'model/{run_name}.pt')
    with open(PATH + f'config/{run_name}.json','w') as f:
        json.dump(vars(config),f)

    datapoints = np.zeros([C,C], dtype=int)
    for x,y in test_dataset:
        datapoints[x[0],x[1]] = 1


    print('distance_dependency', distance_dependency)
    run.summary['distance_dependency'] = distance_dependency
    run.finish()
    return

for num_try in range(3):
    dataset_seed = random.randint(1,99999)
    model_seed = random.randint(1,99999)
    config.dataset_seed = dataset_seed
    config.model_seed = model_seed
    config.epoch = 1000

    for d_model in [128,64]:
        config.d_model = d_model
        for n_heads in [16,8,4,2,1]:
            config.n_heads = n_heads

            run_name = '_'.join([str(dataset_seed), str(d_model), str(n_heads)])
            config.run_name = run_name
            
            train(config)



