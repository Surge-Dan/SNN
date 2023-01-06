import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train

parser = argparse.ArgumentParser(description='MOSI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
# 改：修改TRUE为False，从使用对齐数据到不对齐
parser.add_argument('--aligned', action='store_true', default=False,
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosi',
                    help='dataset to use (default: mosi_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.2,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1,
                    help='output layer dropout')
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=10,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: Adam)')
# 改：原本是40轮epoch
parser.add_argument('--num_epochs', type=int, default=12,
                    help='number of epochs (default: 12)')
parser.add_argument('--when', type=int, default=7,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')
parser.add_argument('--log_interval', type=int, default=10,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()
torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly
print('valid_partial_mode',valid_partial_mode)
if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")
# 改：修改use_cuda把TRUE改为FALSE
use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True
print("Start loading the data....")
train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")
hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = 1
hyp_params.criterion = 'L1Loss'

if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

