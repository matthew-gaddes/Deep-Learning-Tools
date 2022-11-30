#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:27:04 2022

@author: matthew
"""

import tensorflow as tf

#%%


  
class save_all_metrics(tf.keras.callbacks.Callback):
    """ Save the metrics (loss, accuracy etc) for each batch.  
    Inputs:
        metrics | list of strings | the names of the metrics we're intersted in.  
        batch_metrics | dict | key is metric name, value is a list to store that metric for each epoch.  
    Returns:
        
    History:
        2022_11_07 | MEG | Written.  
        2022_11_08 | MEG | Update to return both averaged and non-averaged loss.  

    """
    
    
    def __init__(self, metrics):                                          # metrics is a list of the names of the metrics we want.  Batch metrics is a dict with a list for each metric's value after each batch.  
    
        batch_metrics_mean = {}                                                                         # dict to store metrics and their values for every batch.  Tensorflow >=2.2 default of averaging each batch over the epoch
        batch_metrics = {}                                                                              # dict to store metrics and their values for every batch.  No averaging.  
        val_metrics = {}
        for metrics_name in metrics:
            batch_metrics_mean[metrics_name] = []
            batch_metrics[metrics_name] = []
            val_metrics[f"val_{metrics_name}"] = []
   
        self.metrics = metrics
        self.batch_metrics_mean = batch_metrics_mean
        self.batch_metrics = batch_metrics
        self.val_metrics = val_metrics
    
    def on_epoch_begin(self, batch, logs=None):                                                                             # 
    
        epoch_batch_metrics = {}                                                                # a dict to store the metrics for each batch in each epoch.  
        for metrics_name in self.metrics:
            epoch_batch_metrics[metrics_name] = []                                                                  # fill with empty lists.      
        self.epoch_batch_metrics = epoch_batch_metrics
    
    
    def on_train_batch_end(self, batch, logs=None):                                                                             # batch is just batch number.  
        import numpy as np
        
        for metric in self.metrics:                                                                                             # iterate over the metrics we chose (there may be more in logs)
            self.batch_metrics_mean[metric].append(logs[metric])                                                           # get the mean
            
        for metric in self.metrics:                                                                                             # iterate over the metrics we chose (there may be more in logs)    
            if batch == 0:                                                                                                     # for the first batch...
                self.epoch_batch_metrics[metric].append(logs[metric])                                                                 # which is equal to the first batch loss, as we only have one batch.  
            else:
                batch_loss = ((batch + 1) * logs[metric]) - np.sum(self.epoch_batch_metrics[metric])                                  # rearranging of average equation to find the last value.  batch starts at 0 so add 1 to make it start at 1
                # if batch_loss < 0:
                #     pdb.set_trace()
                self.epoch_batch_metrics[metric].append(batch_loss)

                    
    def on_epoch_end(self, batch, logs=None):                                                                   # every epoch, record the validation loss/accuracy
        for metric in self.metrics:         
            self.val_metrics[f"val_{metric}"].append(logs[f"val_{metric}"])
        
            self.batch_metrics[metric].extend(self.epoch_batch_metrics[metric])
            
            

#%%


class save_model_each_epoch(tf.keras.callbacks.Callback):
        
    def __init__(self, output_path):                                                                    # path to save the model to, including the name of the directory to 
        self.output_path = output_path
    
    def on_epoch_end(self, epoch, logs={}):                                                                 # overwrite the on_epoch_end default metho in the callback class.  
        from pathlib import Path
        print(f"Saving the model at the end of epoch {epoch:03d}")
        path_parts = list(self.output_path.parts)
        path_parts[-1] = f"{path_parts[-1]}_epoch_{epoch:03d}.h5"                                       # append the epoch number to the last part of the path (which is the filename)
        output_path = Path(*path_parts)                                                             # rebuild the path
        self.model.save(output_path)                  # 
        
#%%


class training_figure_per_epoch(tf.keras.callbacks.Callback):
    """ Create a figure showing the training up to that epoch.  
    """
    

    def __init__(self, plotting_function, batch_metrics, epoch_metrics, metrics, title = 'Training metrics',
                 two_column = False, out_path = None, y_epoch_start = 0):
        self.plotting_function = plotting_function
        self.batch_metrics = batch_metrics
        self.epoch_metrics = epoch_metrics
        self.metrics = metrics
        self.title = title
        self.two_column = two_column
        self.out_path = out_path
        self.y_epoch_start =y_epoch_start
    
    def on_epoch_end(self, epoch, logs={}):                                                                 # overwrite the on_epoch_end default metho in the callback class.  
        print(f"Plotting the training figure at the end of epoch {epoch:03d}")
        if epoch == 0: 
            y_epoch_start_temp = 0
        else:
            y_epoch_start_temp = self.y_epoch_start
                    
        self.plotting_function(self.batch_metrics, self.epoch_metrics, self.metrics, f"{self.title} epoch: {epoch:02d}", 
                               self.two_column, self.out_path / f"model_training_epoch_{epoch:02d}.png", y_epoch_start_temp, no_window = True)                    # noet no_window so that figure is just written to outdiretory.  
        
        