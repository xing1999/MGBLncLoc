import os
from pathlib import Path
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
from evaluation import per_eva,evaluate
import pickle
from keras.models import load_model
import numpy as np


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
            chongfu += 1
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, label


def predict(X_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):
    # with open('test_true_label.pkl', 'wb') as f:
    #     pickle.dump(y_test, f)

    adam = Adam(lr=para['learning_rate']) # adam optimizer
    for ii in range(0, len(weights)):
        # 1.loading weight and structure (model)

        # json_file = open('BiGRU_base/' + jsonFiles[i], 'r')
        # model_json = json_file.read()
        # json_file.close()
        # load_my_model = model_from_json(model_json)
        # load_my_model.load_weights('BiGRU_base/' + weights[i])
        # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        h5_model_path = os.path.join(dir, weights[ii])
        load_my_model = load_model(h5_model_path)
        print("Prediction is in progress")

        # 2.predict
        score = load_my_model.predict(X_test)
        # print(score)
        "========================================"
        for i in range(len(score)):
            max_index = np.argmax(score[i])
            for j in range(len(score[i])):
                if j == max_index:
                    score[i][j] = 1
                else:
                    score[i][j] = 0
        # print(score)
        print("Now start testing for Model-{}".format(ii+1))
        label,pred,macro_precision, macro_recall, macro_F1_Score, macro_accuracy,auc=evaluate(score, y_test)
        print("macro_precision:{}  macro_recall:{}  macro_F1_Score:{}  macro_Accuracy:{}  AUC:{} "
              .format(macro_precision, macro_recall, macro_F1_Score, macro_accuracy,auc))
        precision, recall, f1, accuracy_per_class = per_eva(label, pred)
        out = dir
        Path(out).mkdir(exist_ok=True, parents=True)
        out_path2 = os.path.join(out, 'result_test.txt')
        with open(out_path2, 'a+') as fout:
            fout.write('macro_precision:{}\n'.format(macro_precision))
            fout.write('macro_recall:{}\n'.format(macro_recall))
            fout.write('macro_F1_Score:{}\n'.format(macro_F1_Score))
            fout.write('macro_accuracy:{}\n'.format(macro_accuracy))
            fout.write('AUC:{}\n'.format(auc))
            for i in range(4):
                fout.write("label-{}:  Precision:{}  Recall:{}  F1-Score:{}  Accuracy_per:{}\n".format(i, precision[i],
                                                                                                       recall[i], f1[i],
                                                                                                       accuracy_per_class[
                                                                                                           i]))
            fout.write('-------------Test{}-----------\n'.format(ii))
        "========================================"

        # 3.evaluation
        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / len(weights)

    # data saving
    with open(os.path.join(dir, 'MGBLncLoc_prediction_prob.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # getting prediction label
    for i in range(len(score_label)):
        max_index = np.argmax(score_label[i])
        for j in range(len(score_label[i])):
            if j==max_index: score_label[i][j] = 1
            else: score_label[i][j] = 0

    # data saving
    with open(os.path.join(dir, 'MGBLncLoc_prediction_label.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # evaluation
    label,pred,macro_precision, macro_recall, macro_F1_Score, macro_accuracy,auc = evaluate(score_label, y_test)
    print("macro_precision:{}  macro_recall:{}  macro_F1_Score:{}  macro_Accuracy:{}  AUC:{} "
          .format(macro_precision, macro_recall, macro_F1_Score, macro_accuracy, auc))
    precision, recall, f1, accuracy_per_class=per_eva(label,pred)
    out = dir
    Path(out).mkdir(exist_ok=True, parents=True)
    out_path2 = os.path.join(out, 'result_eva.txt')
    with open(out_path2, 'w') as fout:
        fout.write('macro_precision:{}\n'.format(macro_precision))
        fout.write('macro_recall:{}\n'.format(macro_recall))
        fout.write('macro_F1_Score:{}\n'.format(macro_F1_Score))
        fout.write('macro_accuracy:{}\n'.format(macro_accuracy))
        fout.write('AUC:{}\n'.format(auc))
        for i in range(4):
            fout.write("label-{}:  Precision:{}  Recall:{}  F1-Score:{}  Accuracy_per:{}\n".format(i, precision[i], recall[i],f1[i], accuracy_per_class[i]))
        fout.write('\n')



def test_my(test, para, model_num, dir):
    # step1: preprocessing
    test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp

    # weight and json
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num+1):
        weights.append('model{}.h5'.format(str(i)))
        jsonFiles.append('model{}.json'.format(str(i)))

    # step2:predict

    predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir)
