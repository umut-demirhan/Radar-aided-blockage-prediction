"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Evaluation script for radar aided blockage prediction task
Author: Umut Demirhan
Date: 10/29/2021
"""

from dataset import load_radar_data
import numpy as np
from tensorflow import keras
import radar_preprocessing
from time_series import TimeSequenceGenerator
import re
import os

# Root directory of the development dataset
root_dir = r'.\scenario30\development_dataset'

# CSV file of all single samples
csv_file = 'scenario30_single_full.csv'

# CSV file of test sequences
csv_test = 'scenario30_series_test.csv'

#%% DATA LOAD

X_all, y_all = load_radar_data(root_dir, csv_file, radar_column='unit1_radar_1', label_column='blockage_1')

preprocessing_fn = getattr(radar_preprocessing, 'range_angle_map')

X_all = preprocessing_fn(X_all)

#%%
# The NN models are provided for the blockage interval values 1-10
blockage_interval = np.arange(1, 11)
results = [] # F1 scores - Accuracy

for interval in blockage_interval:
    
    model_name = './saved_models/blockageinterval%i_seqsize8_LSTMsize64' % interval


    parameters = re.findall('\d+', model_name)
    blockage_duration = int(parameters[0])
    sequence_size = int(parameters[1])
    LSTM_size = int(parameters[2])
    
    
    series_csv_path = os.path.join(root_dir, csv_test)
    timeseries_test = TimeSequenceGenerator(x=X_all, 
                                            y=None,
                                            len_seq_x=sequence_size, 
                                            len_seq_y=blockage_duration, 
                                            batch_size=None, 
                                            shuffle=False, 
                                            csv_file=series_csv_path
                                           )
    
    model = keras.models.load_model(model_name)
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
    from tqdm import tqdm
    
    acc = 0
    y_hat = []
    y = []
    for i in tqdm(range(len(timeseries_test))):
        x_batch, y_batch = timeseries_test[i]
        acc += accuracy_score(model.predict(x_batch)>0.5, y_batch)
        y_hat.append(model.predict(x_batch)>0.5)
        y.append(y_batch)
    acc /= len(timeseries_test)
    
    y_hat = np.concatenate(y_hat).flatten()
    y = np.concatenate(y).flatten()
    np.sum(np.equal(y_hat, y))/len(y_hat)
    
    f1_s = f1_score(y, y_hat)
    acc_s = accuracy_score(y, y_hat)
    print('Blockage interval - %i blocks' % interval)
    print('Accuracy: %.8f' % acc_s)
    print('F1 score: %.8f' % f1_s)
    print('-'*40)
    results.append([f1_s, acc_s])
    
results = np.array(results)
print(results)

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(blockage_interval*110, results, '-x')
plt.legend(['F1 Score', 'Accuracy'])
plt.grid()
plt.xlabel('Blockage interval (ms)')