from argparse import ArgumentParser
from training import train_model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=0)
    parser.add_argument('--pre_trained', type=str, default=None)
    parser.add_argument('--label_dict', type=str, default='./data/y2_id_dic.txt')

    # parser.add_argument('--train_dataset', type=str, default='data/train.json')
    # parser.add_argument('--dev_dataset', type=str, default='data/dev.json')

    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=25)
    parser.add_argument('--dict_number', type=int, default=21128)
    parser.add_argument('--num_labels', type=int, default=53)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)

    #parser.add_argument('--save_model', type=str, default='models/bert.pt')
    option = parser.parse_args()

    train_model(option)