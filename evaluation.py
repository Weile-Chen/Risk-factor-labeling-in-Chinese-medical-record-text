from utils.conlleval import evaluate
import pickle


class evaluation():
    def __init__(self,label_dict_path):
        self.y_pred = []
        self.y_true = []
        with open(label_dict_path, 'rb') as f:
            self.label_dict = pickle.load(f)

    def add(self, y_pred_, batch_y_ture):
        self.y_pred += y_pred_
        self.y_true += batch_y_ture.numpy().tolist()

    def evaluate(self):
        # 压平到一维
        self.y_pred = sum(self.y_pred,[])
        self.y_true = sum(self.y_true, [])
        print(self.y_true)
        # 转换对应的label
        self.y_pred = [self.label_dict[int(i)] for i in self.y_pred]
        self.y_true = [self.label_dict[int(i)] for i in self.y_true]

        return evaluate(self.y_true, self.y_pred, verbose=False)