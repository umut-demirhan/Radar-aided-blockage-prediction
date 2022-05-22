# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:20:36 2021

@author: Umt
"""

import numpy as np
import tensorflow.keras as keras
import pandas as pd


class TimeSequenceGenerator(keras.utils.Sequence):
    def __init__(self, x, y, len_seq_x=8, len_seq_y=3, batch_size=1, shuffle=False, csv_file='scenario30_series_train.csv'):
        self.len_seq_x = len_seq_x
        self.len_seq_y = len_seq_y
        
        self.csv_file = csv_file

        self.x = x
        self.y = y
        self.shuffle = shuffle
        
        self._load_sequences()
        
        self.batch_size = self.size if batch_size is None else batch_size
        
        self.seq_index = np.arange(self.size)
        
        self.on_epoch_end()

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, index):
        seq_index_batch = self.seq_index[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.x[self.sequence_list_x[seq_index_batch, -self.len_seq_x:]-1]
        Y = np.any(self.sequence_list_y[seq_index_batch, :self.len_seq_y], axis=1)
        
        return X, Y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.seq_index)

    def _load_sequences(self):
        self.csv_frame = pd.read_csv(self.csv_file)
        
        x_count = 0
        y_count = 0
        for column_name in self.csv_frame.columns:
            if 'x_' in column_name:
                x_count +=1
            if 'blockage_' in column_name:
                y_count += 1
        
        self.csv_frame.columns
        self.sequence_list_x = np.empty((0, x_count), int)
        self.sequence_list_y = np.empty((0, y_count), int)
        
        for i in range(len(self.csv_frame)):
            x_seq = np.array([self.csv_frame['x_%i'%j][i] for j in np.arange(x_count)+1])
            y_seq = np.array([self.csv_frame['blockage_%i'%j][i] for j in np.arange(y_count)+1])
            self.sequence_list_x = np.vstack((self.sequence_list_x, x_seq))
            self.sequence_list_y = np.vstack((self.sequence_list_y, y_seq))
        
        self.x_count = x_count
        self.y_count = y_count
               
        self.size = self.sequence_list_x.shape[0]
        
    def shuffle_indices(self):
        np.random.shuffle(self.seq_index)
 
    
 
# x = TimeSequenceGenerator(np.arange(20000), -np.arange(20000), csv_file=r'C:\Users\udemirha\Desktop\scenario30\development\scenario30_series_train.csv')


# for i in range(10):
#     x1, y1 = x[i]
#     print(x1)
#     print(y1)


