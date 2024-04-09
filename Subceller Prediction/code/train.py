import os
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from model import base, BiGRU_base,MG_BiGRU_base
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from evaluation import convert_to_labels,evaluate,per_eva
import numpy as np
np.random.seed(101)
from pathlib import Path



def catch(data, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            # print(t)
            chongfu += 1
            # print(data[t[0]])
            # print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, label





def train_my(train, para, model_num, model_path):

    Path(model_path).mkdir(exist_ok=True)

    # data get
    X_train, y_train = train[0], train[1]

    # data and label preprocessing \Bin
    y_train = keras.utils.to_categorical(y_train)
    X_train, y_train = catch(X_train, y_train)
    y_train[y_train > 1] = 1

    # disorganize
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    # train
    length = X_train.shape[1]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon,t_data.tm_mday,t_data.tm_hour,t_data.tm_min,t_data.tm_sec))


    for counter in range(1, model_num+1):
        print("Now start training for Model-{}".format(counter))
        # get my neural network model
        if model_path == 'base':
            model = base(length, out_length, para)
        elif model_path == 'BiGRU_base':
            model = BiGRU_base(length, out_length, para)
        elif model_path == 'MG_BiGRU_base':
            model = MG_BiGRU_base(length, out_length, para)
        else:
            print('no model')
        model.fit(X_train, y_train, nb_epoch=2, batch_size=64, verbose=2)
        score = model.predict(X_train)
        for i in range(len(score)):
            max_index = np.argmax(score[i])
            for j in range(len(score[i])):
                if j == max_index:
                    score[i][j] = 1
                else:
                    score[i][j] = 0
        # print(score)
        # print(predictions_train)
        label, pred, macro_precision, macro_recall, macro_F1_Score, macro_accuracy, auc = evaluate(score, y_train)
        print("macro_precision:{}  macro_recall:{}  macro_F1_Score:{}  macro_Accuracy:{}  AUC:{} "
              .format(macro_precision, macro_recall, macro_F1_Score, macro_accuracy, auc))
        # print(labels)
        precision, recall, f1, accuracy_per_class = per_eva(label, pred)
        each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter),tt.tm_mon,tt.tm_mday,tt.tm_hour,tt.tm_min,tt.tm_sec))



import time
from test import test_my
def train_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    train_my(train, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test start time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

    # test_my(test, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
