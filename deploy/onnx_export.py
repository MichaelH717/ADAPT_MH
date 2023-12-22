import torch

import argparse
import os
from torch.utils.data import DataLoader

from read_args import get_args
from model.adapt import ADAPT
from dataset.argoverse_dataset import Argoverse_Dataset, batch_list

parser = argparse.ArgumentParser("onnx_export arguments")

# === Data Related Parameters ===
parser.add_argument('--checkpoint_path', type=str, default='/home/movex/AI/ADAPT_MH/checkpoint/exp1/checkpoint.pt')
parser.add_argument('--onnx_path', type=str, default='/home/movex/AI/ADAPT_MH/deploy/prediction_onnx')
args1 = parser.parse_args()

if __name__ == '__main__':
    rank = torch.device("cuda")
    assert os.path.exists(args1.checkpoint_path)
    checkpoint = torch.load(args1.checkpoint_path)

    # load checkpoint
    args = get_args()
    model = ADAPT(args)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(rank)

    # input data
    train_dataset = Argoverse_Dataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  pin_memory=False, drop_last=False, shuffle=False,
                                  collate_fn=batch_list)

    train_dataloader_iter = iter(train_dataloader)
    dataset_input = train_dataloader_iter.__next__()

    # onnx
    onnx_path = args1.onnx_path
    model.eval()
    torch.onnx.export(model, (dataset_input,), onnx_path, input_names=["input_map"], output_names=["output_map"])
    print('export onnx done')
