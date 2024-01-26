import argparse
parser = argparse.ArgumentParser()

### dataset
parser.add_argument('--n_vocab', default=59, type=int, help='vocabulary size of numbers')
parser.add_argument('--frac', default=0.7, type=float, help='training set fraction')
parser.add_argument('--abel', default=False, type=bool, help='whether the problem is an Abel group')
# parser.add_argument('--dataset_seed', default=310, type=int, help='training set fraction')

### model
parser.add_argument('--d_model', default=128, type=int, help='embedding dimension')
parser.add_argument('--tied_embeddings', default=False, type=bool, help='share the encoding and decoding')
    # linear modal
# parser.add_argument('--model_type', default='A', type=str, help='the specific type of the model')
    # transformer
parser.add_argument('--n_heads', default=1, type=int, help='number of self-attention heads')
parser.add_argument('--n_layers', default=1, type=int, help='number of layers')
parser.add_argument('--act_fn', default='gelu', type=str, help='active function')
# parser.add_argument('--use_linear', default=False, type=bool, help='active function')

### training
parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1., type=float)

### display
parser.add_argument('--tqdm_disable', default=False, type=bool, help='disable tqdm or not')

cfg = parser.parse_args('')




