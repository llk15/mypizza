import numpy as np
import torch
import torch.optim as optim
import random
import json

import wandb
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

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

# train
def train(config, dataset_seed, model_seed):
    # dataset
    torch.manual_seed(dataset_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=C*C, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=C*C, shuffle=False)
    # model
    models={'A':MyModelA,'B':MyModelB,'C':MyModelC,'D':MyModelD,'X':MyModelX}
    torch.manual_seed(model_seed)
    model = models[config.model_type](n_vocab=config.n_vocab, d_hidden=config.d_model)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = cross_entropy_high_precision

    run = wandb.init(reinit=True, name=config.run_name, config=vars(config), project='grokking', group = str(config.dataset_seed) )#,settings=wandb.Settings(start_method="spawn"))
    bar = tqdm(range(config.epoch), bar_format='{l_bar}{bar:30}{r_bar}', mininterval=1.0, disable=config.tqdm_disable)
    for epoch in bar:
        # train
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss=loss.item()
        log = ('epoch %d [TRAIN] loss: %.3g ' % (epoch + 1, loss.item()))
        # test
        with torch.no_grad():
            total = total_loss = total_correct = 0
            for data in test_loader:
                inputs, labels = map(lambda t:t.to(DEVICE),data)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            val_loss=total_loss/len(test_loader)
            val_acc=total_correct/total
            cur_norm=norm(model)
            log += ('[TEST] loss: %.3g accuracy: %.3f [MODEL] norm: %.3f' % (total_loss/len(test_loader), total_correct/total, norm(model)))
        bar.set_description(log, refresh=False)
        # wandb
        if run:
            run.log({'train_loss': train_loss,
            'test_loss': val_loss,
            'test_accuracy': val_acc,
            'norm': cur_norm})

    # evaluate
    y_logits = np.zeros([C,C]) # logits for target logits
    ab_logits = np.zeros([C,C]) # output(A+B)_output(A-B) matrix for target logits
    for x,y in full_loader:
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        with torch.inference_mode():
            model.eval()
            logits = model(x) # [num_sample, C]
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

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # sns.heatmap(1-datapoints, ax=ax1)
    # ax1.set_title("correct logits")
    # sns.heatmap(y_logits, ax=ax2)
    # ax2.set_title("top 1 logits")
    # fig.savefig(PATH + f'png/{run_name}.png')

    # logging and return
    print('distance_dependency', distance_dependency)
    run.summary['distance_dependency'] = distance_dependency
    run.finish()
    return

for num_try in range(12):
    # random setting
    dataset_seed = random.randint(1,99999)
    model_seed = random.randint(1,99999)
    config.dataset_seed = dataset_seed
    config.model_seed = model_seed
    for model_type in 'ABCDX':
        config.model_type = model_type
        for weight_decay in [1e-0, 1e-1, 1e-2, 1e-3]:
            config.weight_decay = weight_decay

            run_name = '_'.join([str(dataset_seed), str(model_seed), str(model_type), str(weight_decay)])
            config.run_name = run_name
            print(run_name)

            train(config, dataset_seed, model_seed)