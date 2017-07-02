print('\n >>>>>>>> Dev new stat2892\n')

import argparse
trained_model_path = '../trained_models/last_dnn.h5'
parser = argparse.ArgumentParser(
    description='DNNs on Keras to predict conversion rate.')
parser.add_argument('--et', type=str, default=None, 
    help='evaluate trained model (the last run by default))')
parser.add_argument('--ns', action='store_true', 
    help='don\'t save model in the end')
parser.add_argument('--of', action='store_true', 
    help='use only one feature')
parser.add_argument('--nm', action='store_true', 
    help='not mask future information')
parser.add_argument('-e', type=int, default=5,
    help='epochs')
parser.add_argument('-f', type=int, default=24,
    help='the f most important independent features (<=24)')
parser.add_argument('-v', type=int, default=1,
    help='verbose')
parser.add_argument('-b', type=int, default=1048,
    help='batch size')
parser.add_argument('--opt', type=str, default='adagrad',
    help='batch size')
parser.add_argument('--mt', type=int, default=3,
    help='model type')
parser.add_argument('--va-seed', type=int, default=62,
    help='numpy random seed number to split tr and val')
parser.add_argument('--va', type=float, default=0,
    help='split validation from train')
parser.add_argument('-s', action='store_true',
    help='print model summary')
parser.add_argument('--ne', action='store_true',
    help='use native embedding')
parser.add_argument('--nfe', action='store_true',
    help='not use fine-grained embedding layers')
parser.add_argument('--ct', action='store_true',
    help='continue training last model')
parser.add_argument('--mess', type=str, default='no_mess',
    help='leave a message')
parser.add_argument('-m', type=str, default='mlp', required=True,
                    help='model name - lr | mlp | mlp_fe | elr | rf | xgb')
parser.add_argument('--emb', type=str, default='n_vd', required=True,
                    help='embedding mode - n_fd | n_vd | s ')


g = parser.add_mutually_exclusive_group(required=False)
g.add_argument('-tdo', action='store_true',
    help='two days only (17 and 24)')
g.add_argument('-rml', action='store_true',
    help='remove day 30')
g.add_argument('-fra', action='store_true',
    help='take fraction of data')
g.add_argument('-olv', action='store_true',
    help='use offline validation train and test datasets')


args = parser.parse_args()
print(args)
import os
import sys
import shutil
import json
import datetime
import math
import functools
import pickle
import numpy as np
import pandas as pd
from time import time, strftime
from collections import Counter
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model, Sequential, Model
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input, Reshape
from keras.layers.merge import Concatenate, Add, concatenate, add
from keras.optimizers import rmsprop, sgd, adam, adagrad
os.system('setterm -cursor off')

def s_c(x):
    return [x[:, i:i+1] for i in range(len(x[0]))]

def save_preds(preds, cb=False):
    preds = np.ravel(preds)
    assert len(preds)==338489
    avg = np.average(preds)
    std = np.std(preds)
    p = '../%s%s_tl_result_%.4f_%.4f_%s.csv' % ('cb_' if cb else '', 
        strftime('%H%M_%m%d'), avg, std, args.m)

    df = pd.DataFrame({'instanceID': te_df_['instanceID'].values, 'proba': preds})
    df.sort_values('instanceID', inplace=True)
    df.to_csv(p, index=False)

    if cb: 
        return avg, std, p
    else:
        print('\nTrain average: ', tr_avg)
        print('Preds average: ', avg)
        print('Preds std dev.: ', std)
        print('\nWritten to: ', p)


class predCallback(Callback):
    def __init__(self, test_data):
        self.te_x = test_data

    def on_epoch_end(self, epoch, logs={}):
        predict_probas = np.ravel(self.model.predict(self.te_x, batch_size=40960, verbose=1))
        avg, std, p = save_preds(predict_probas, cb=True)
        print('\nTr avg: %.4f,   avg: %.4f, std: %.4f, written to: %s\n'%(tr_avg, avg, std, p))



def batch_generator(x, y=None, batch_size=args.b):
    nb_batch = np.ceil(len(x)/batch_size)
    if y is None:
        for xb in np.array_split(x, nb_batch):
            yield s_c(xb)
    else:
        for xb,yb in zip(np.array_split(x, nb_batch), np.array_split(y, nb_batch)):
            yb = to_categorical(yb, 110527)
            yield s_c(xb), yb

# ======================================================================================================== #
# 1 data
# ======================================================================================================== #
te = pd.read_csv('../data/test_new.csv')
tr = pd.read_csv('../data/train_new.csv')
va = tr[tr.d >= 22]
tr = tr.drop(va.index)
tr = tr.sample(frac=1)

features = ['userid', 'bikeid', 'biketype', 'wd', 'd', 'h', 'm', 'start']

tr_x = tr[features].values
va_x = va[features].values
te_x = te[features].values
tr_y = tr.end.values
va_y = va.end.values

# tr_y = to_categorical(tr_y, 110527)
# va_y = to_categorical(va_y, 110527)

# ======================================================================================================== #
# 2 model
# ======================================================================================================== #
max_fs = pd.concat([tr[features], te[features]]).max().values +1

ins = [Input(shape=(1, )) for _ in range(len(features))]
cols_outs = []
for i,inp in enumerate(ins):
    out = Embedding(max_fs[i], 16)(inp)
    out = Flatten()(out)
    cols_outs.append(out)

cols_out = concatenate(cols_outs)
if args.mt == 0:
    y = Dense(1024, activation='relu', 
              kernel_regularizer='l1')(cols_out)
    y = Dropout(.3)(y)
    y = Dense(512, activation='relu')(y)

if args.mt == 1:
    y = Dense(1024, activation='relu', 
              kernel_regularizer='l1')(cols_out)
    y = Dropout(.2)(y)
    y = Dense(512, activation='relu')(y)

if args.mt == 2:
    y = Dense(1024, activation='relu', 
              kernel_regularizer='l1')(cols_out)
    y = Dropout(.1)(y)
    y = Dense(512, activation='relu')(y)

if args.mt == 3:
    y = Dense(1024, activation='relu')(cols_out)
    y = Dropout(.1)(y)
    y = Dense(1024, activation='relu')(y)

if args.mt == 4:
    y = Dense(1024, activation='relu')(cols_out)
    y = Dense(512, activation='relu')(y)
    y = Dense(512, activation='tanh')(y)

if args.mt == 5:
    y = cols_out

y = Dense(110527, activation='softmax')(y)  
model = Model(ins, y)
model.summary()

top3_acc = functools.partial(top_k_categorical_accuracy, k=3)

model.compile(optimizer='adagrad', 
              loss='categorical_crossentropy', 
              # metrics=[top3_acc, 'categorical_crossentropy']
              )
# ======================================================================================================== #
# 4 train
# ======================================================================================================== #

# model.fit(tr_x, tr_y, epochs=args.e, batch_size=args.b,
#         validation_data=(va_x, va_y),
#         shuffle=True,
#         )
print('\n\nStart training')
len_tr = len(tr_y)
for e in range(args.e):
    trained = 0
    print('\n\n--------- Epoch ', e)
    start = time()
    for tr_xyb in batch_generator(tr_x, tr_y):
        logs = model.train_on_batch(*tr_xyb)
        trained += len(tr_xyb[1])
        print('%d/%d\t - Time: %ds\t - Loss: %.4f'%(trained, len_tr, int(time()-start), logs), end='\r')
        # sys.stdout.flush()

        # if trained > 10000: 
        #     print(trained)
        #     break

model.save('../trained_models/%smobai_model.h5' % strftime('%H%M_%m%d_'))

# ======================================================================================================== #
# 4 test
# ======================================================================================================== #
le = pickle.load(open('../data/laberEncoder', 'rb'))

print('\nTesting')
len_te = len(te_x)
idxs = None
tested = 0
start = time()
for te_xb in batch_generator(te_x, batch_size=1024*16):
    tested += len(te_xb[0])
    preds_b = model.predict_on_batch(te_xb)
    idxs_b = (-preds_b).argsort()[:, :3]
    idxs = idxs_b if idxs is None else np.vstack((idxs, idxs_b))
    print('Tested %d/%d\t - Time: %ds'%(tested, len_te, int(time()-start)), end='\r')
    sys.stdout.flush()
    
submission = pd.DataFrame(idxs).apply(le.inverse_transform)

# assert len(submission) == len(te)

submission['orderid'] = te.orderid
print(submission.head())

submission = submission[['orderid', 0, 1, 2]]

submission.to_csv('../../results/%s_submission'%strftime('%H%M_%m%d'), header=None, index=None)


# ======================================================================================================== #
os.system('setterm -cursor on')
