from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout,Conv1D,Add, Dense,TimeDistributed
from keras.layers import Flatten, Dense, Activation, BatchNormalization, CuDNNGRU, CuDNNLSTM, Lambda,GRU,Multiply,Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.regularizers import l2
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import Reshape
import tensorflow as tf
import numpy as np

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))
    ))
    return x * cdf
def mdta_module(x):
    x = BatchNormalization()(x)
    filters = 32
    x = Conv1D(filters=filters, kernel_size=1, activation='linear')(x)
    x = Conv1D(filters=filters, kernel_size=3, padding='same', activation='linear')(x)
    x_flat = Flatten()(x)
    q = x_flat
    k = x_flat
    v = x_flat
    attention_map = Lambda(lambda x: K.softmax(K.batch_dot(x[0], x[1], axes=(1, 1))))([q, k])
    attended_values = Lambda(lambda x: K.batch_dot(x[0], K.expand_dims(x[1], -2)))([attention_map, v])
    reshaped_attended_values = Reshape((-1, filters))(attended_values)
    output = Lambda(lambda x: x[0] + x[1])([x, reshaped_attended_values])
    return output

def gdfn_module(x):
    x_bn = BatchNormalization()(x)
    filters = 32
    h_pw = Conv1D(filters=filters, kernel_size=1, activation='linear')(x_bn)
    i_pw = Conv1D(filters=filters, kernel_size=1, activation='linear')(x_bn)
    h = Conv1D(filters=filters, kernel_size=3, padding='same', activation='linear')(h_pw)
    i = Conv1D(filters=filters, kernel_size=3, padding='same', activation='linear')(i_pw)
    h = Activation(gelu)(h)
    h_i = Multiply()([h, i])
    x_gdfn = Conv1D(filters=filters, kernel_size=1, activation='linear')(h_i)
    x_gdfn = Lambda(lambda x: x[0] + x[1])([x_bn, x_gdfn])
    return x_gdfn

def MG_BiGRU_base(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001
    main_input = Input(shape=(length,), dtype='float32', name='main_input')
    x = Reshape((length, 1))(main_input)  # Reshape layer added here

    x_mdta = Lambda(lambda x: mdta_module(x), name='mdta_module')(x)
    x_gdfn = Lambda(lambda x: gdfn_module(x), name='gdfn_module')(x_mdta)

    main_input_combined = Add()([x_gdfn, x_mdta])

    # a = Conv1D(64, kernel_size=2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x_mdta)
    # a = Conv1D(64, kernel_size=2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x_gdfn)

    a = Conv1D(64, kernel_size=2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(main_input_combined)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    b = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(main_input_combined)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    c = Conv1D(64, kernel_size=8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(main_input_combined)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)
    x = Bidirectional(GRU(50, return_sequences=True))(merge)
    x = Flatten()(x)
    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001
    main_input = Input(shape=(length,), dtype='float32', name='main_input')
    x = Reshape((length, 1))(main_input)
    a = Conv1D(64, kernel_size=2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    b = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    c = Conv1D(64, kernel_size=8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)
    x = Flatten()(merge)
    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def BiGRU_base(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='float32', name='main_input')
    x = Reshape((length, 1))(main_input)
    a = Conv1D(64, kernel_size=2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)

    b = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)

    c = Conv1D(64, kernel_size=8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Bidirectional(GRU(50, return_sequences=True))(merge)
    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    # output = Dense(out_length, activation='softmax', name='output', kernel_regularizer=l2(l2value))(x)
    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

