import pickle
import random


with open('./x_id.txt', 'rb') as f:
    x = pickle.load(f)
with open('./y2_id.txt', 'rb') as f:
    y = pickle.load(f)

randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(x)
random.seed(randnum)
random.shuffle(y)

with open('./train_x.pkl', 'wb') as f:
    pickle.dump(x[:int(len(x)*(7/10))], f)
with open('./test_x.pkl', 'wb') as f:
    pickle.dump(x[int(len(x)*(7/10)):], f)

with open('./train_y.pkl', 'wb') as f:
    pickle.dump(y[:int(len(x)*(7/10))], f)
with open('./test_y.pkl', 'wb') as f:
    pickle.dump(y[int(len(x)*(7/10)):], f)