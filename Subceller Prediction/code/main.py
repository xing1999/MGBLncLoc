import time
import numpy as np
from pathlib import Path
from  encode import load_data,load_test_data
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# dir = 'BiGRU_base'
# dir="base"
dir="MG_BiGRU_base"
Path(dir).mkdir(exist_ok=True)
t = time.localtime(time.time())
with open(os.path.join(dir, 'time.txt'), 'w') as f:
    f.write('start time: {}m {}d {}h {}m {}s'.format(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    f.write('\n')


def GetSequenceData():

    tr_file_path =".\\Data\\Train_101.tsv"
    tr_label,traindata,MCD = load_data(tr_file_path)

    # tr_label, traindata = [],[]
    te_file_path = ".\\Data\\test_101.tsv"
    te_label, testdata = load_test_data(te_file_path,MCD)
    # data type conversion
    train_data = np.array(traindata)
    test_data = np.array(testdata)
    train_label = np.array(tr_label)
    test_label = np.array(te_label)

    return [train_data, test_data, train_label, test_label]

def GetData():
    # get sequence data
    sequence_data = GetSequenceData()
    return sequence_data

def TrainAndTest(tr_data, tr_label, te_data, te_label):

    from train import train_main # load my training function

    train = [tr_data, tr_label]
    test = [te_data, te_label]

    threshold = 0.5
    model_num = 9  # model number
    test.append(threshold)
    train_main(train, test, model_num, dir)

    ttt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))


def main():
    # I.get sequence data
    sequence_data = GetData()
    # sequence data partitioning
    tr_seq_data,te_seq_data,tr_seq_label,te_seq_label = \
        sequence_data[0],sequence_data[1],sequence_data[2],sequence_data[3]
    # II.training and testing
    TrainAndTest(tr_seq_data, tr_seq_label, te_seq_data, te_seq_label)

if __name__ == '__main__':
    # executing the main function
    main()




