import pandas as pd
import numpy as np
import random
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from recurrent import LSTMSNPCell
from matplotlib import pyplot
import openpyxl
import os
import tensorflow as tf
from keras.layers import Layer

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.keras.backend.clear_session()

train_data = pd.read_excel("train.xlsx")
test_data = pd.read_excel("test.xlsx")

origin_train_x = train_data.iloc[:, 2:].values
origin_train_y = train_data.iloc[:, 1].values
origin_test_x = test_data.iloc[:, 2:].values
origin_test_y = test_data.iloc[:, 1].values

index_train = [j for j in range(len(origin_train_x))]
index_test = [j for j in range(len(origin_test_x))]

random.shuffle(index_train)
random.shuffle(index_test)

origin_train_y = origin_train_y[index_train]
origin_train_x = origin_train_x[index_train]
origin_test_y = origin_test_y[index_test]
origin_test_x = origin_test_x[index_test]

input_size = 18
time_step = 10
labels = 2
epochs = 1000
batch_size = 32
cell = 128

def label2hot(labels):
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(labels.reshape(-1, 1))
    return onehot_encoded

hot_data_y = label2hot(origin_train_y)

scaler = preprocessing.StandardScaler().fit(origin_train_x)
train_x = scaler.transform(origin_train_x)
train_x = train_x.reshape([-1, input_size, time_step])
train_x = np.transpose(train_x, [0, 2, 1])
train_y = hot_data_y

test_x = scaler.transform(origin_test_x)
test_x = test_x.reshape([-1, input_size, time_step])
test_x = np.transpose(test_x, [0, 2, 1])
test_y = label2hot(origin_test_y)

def print_current_time_and_best_metrics(epoch, logs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"time: {current_time}")

print_time_and_best_metrics_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print_current_time_and_best_metrics(epoch, logs)
)

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

input_shape = (time_step, input_size)

input_list = []
lstm_list = []
for i in range(10):
    input_temp = Input(shape=input_shape)
    input_list.append(input_temp)
    lstm_temp = LSTMSNPCell(cell, activation='relu')(input_temp)
    lstm_list.append(lstm_temp)

merged = concatenate(lstm_list)
output = Dense(labels, activation='softmax')(merged)
model = Model(inputs=input_list, outputs=output)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit([train_x] * 10, train_y,
                    validation_data=([test_x] * 10, test_y),
                    epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True,
                    callbacks=[print_time_and_best_metrics_callback, early_stopping, model_checkpoint])

test_loss, test_accuracy = model.evaluate([test_x] * 10, test_y, verbose=2)
train_loss, train_accuracy = model.evaluate([train_x] * 10, train_y, verbose=2)

print('Test accuracy:', test_accuracy)
print('Train accuracy:', train_accuracy)
print('epochs=', epochs, '\nbatch_size=', batch_size)

prediction_label = model.predict([test_x] * 10)
prediction_label_binary = np.argmax(prediction_label, axis=1)
test_y_binary = np.argmax(test_y, axis=1)

TP = np.sum(np.logical_and(prediction_label_binary == 1, test_y_binary == 1))
FP = np.sum(np.logical_and(prediction_label_binary == 1, test_y_binary == 0))
TN = np.sum(np.logical_and(prediction_label_binary == 0, test_y_binary == 0))
FN = np.sum(np.logical_and(prediction_label_binary == 0, test_y_binary == 1))

accuracy = (TP + TN) / (TP + FP + TN + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
f1_score = (2 * TP) / (2 * TP + FP + FN)

print('TP', TP)
print('FP', FP)
print('TF', TN)
print('FN', FN)
print('accuracy=', accuracy)
print('sensitivity=', sensitivity)
print('specificity=', specificity)
print('f1_score=', f1_score)

with open('output_result.txt', 'r') as file:
    lines = file.readlines()

run_count = len([line for line in lines if "time:" in line]) + 1

with open('output_result.txt', 'a') as file:
    file.write(f" {run_count} :\n")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(f"time: {current_time}\n")
    file.write("result: \n")
    file.write("Test accuracy: {}\n".format(train_accuracy))
    file.write("Train accuracy: {}\n".format(test_accuracy))
    file.write("TP: {}\n".format(TP))
    file.write("FP: {}\n".format(FP))
    file.write("TN: {}\n".format(TN))
    file.write("FN: {}\n".format(FN))
    file.write("accuracy: {}\n".format(accuracy))
    file.write("sensitivity: {}\n".format(sensitivity))
    file.write("specificity: {}\n".format(specificity))
    file.write("f1_score: {}\n".format(f1_score))
    file.write("epochs:{}\n".format(epochs))
    file.write("batch_size:{}\n".format(batch_size))
    file.write("cell:{}\n".format(cell))
    file.write("=" * 40 + "\n")

test_x_list = [test_x] * 10
prediction_probabilities = model.predict(test_x_list)
prediction_label = np.argmax(prediction_probabilities, axis=1)
prediction_label = [i + 1 for i in prediction_label]
fact_label = np.argmax(test_y, 1)
fact_label = [i + 1 for i in fact_label]
analysis = [fact_label, prediction_label]
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = 'analysis_data'
for i in range(0, 2):
    for j in range(0, len(analysis[i])):
        sheet.cell(row=j + 1, column=i + 1, value=analysis[i][j])
wb.save('./datas/analysis_label.xlsx')

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize=12)
pyplot.ylabel('Loss', fontsize=12)
pyplot.savefig("./images/Loss_label.png")
pyplot.show()

pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize=12)
pyplot.ylabel('Accuracy', fontsize=12)
pyplot.savefig("./images/Accuracy_label.png")
pyplot.show()
