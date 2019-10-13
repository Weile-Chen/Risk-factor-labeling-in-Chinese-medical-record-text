import torch
from dataset import build_dataset
from torch.utils.data import DataLoader
from biLSTM import biLSTM_CRF
from evaluation import evaluation

def train_model(option):
    train_loader = DataLoader(dataset=build_dataset('./data/train_x.pkl', './data/train_y.pkl'), batch_size=option.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=build_dataset('./data/test_x.pkl', './data/test_y.pkl'), batch_size=option.batch_size, shuffle=True)

    model = biLSTM_CRF( option.embedding_size, option.hidden_size, option.dict_number, option.num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=option.lr)
    if option.use_gpu:
        model.cuda()
    if option.pre_trained:
        model.load_state_dict(torch.load(option.pre_trained))

    for epoch in range(option.epochs):
        print(epoch)
        # 创建评价
        train_eva = evaluation(option.label_dict)
        test_eva = evaluation(option.label_dict)
        # 训练
        model.train()
        for step, (batch_x, batch_y, batch_masks) in enumerate(train_loader):
            optimizer.zero_grad()

            if option.use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                masks = masks.cuda()

            y_pred, loss = model(batch_x, batch_y, batch_masks)
            train_eva.add(y_pred, batch_y)
            loss.backward()
            optimizer.step()
        # 通过测试集验证
        model.eval()
        for step, (batch_x, batch_y) in enumerate(test_loader):
            if option.use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                masks = masks.cuda()

            y_pred, loss = model(batch_x, batch_y, batch_masks)
            test_eva.add(y_pred, batch_y)
        print("train:")
        print(train_eva.evaluate())
        print("test:")
        print(test_eva.evaluate())