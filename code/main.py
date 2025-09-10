import argparse

from train import train
from data_processor import processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # preprocess
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--max_window_num', type=int, default=24)
    parser.add_argument('--input_folder_path', type=str, default='../dataset/final_raw')
    parser.add_argument('--output_folder_path', type=str, default='../dataset/final_npy')
    # data
    parser.add_argument("--data_dir", type=str, default="../dataset/final_npy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    # model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_cls", action="store_true")  # 없으면 mean pooling
    # train
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    processor(args)
    train(args)
