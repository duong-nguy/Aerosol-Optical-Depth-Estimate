from argparse import ArgumentParser
from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--sainity_check", type=bool, default=False)
    args = parser.parse_args()
    train(args)

if __name__=="__main__":
    main()

