import evalTest
import torch

def main():
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    parser.add_argument('-path', type=str, default='', help='path to data file [default:]')
    args = parser.parse_args()

    train_dev = open(train_dev_path).readlines()

    model = torch.load()
    evalTest(train_dev)

if __name__ == '__main__':
    main()
