#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:17:53 2021

@author: matthew
"""

from tensorflow import keras 
import tensorflow as tf

#%%


class numpy_sequence(tf.keras.utils.Sequence):                                                                                  # inheritance not tested like ths.  
        
    def __init__(self, file_list, batch_size):                                          # constructor
        """
        Inputs:
            file_list | list of strings or paths | locations of numpy files of data.  
            batch_size | int | number of data for each batch.  Note tested if larger than the number of data in a single file.  
        """
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):                                                      # number of batches in an epoch
        """As one large file (e.g. 1000 data) can't be used as a batch on a GPU (but maybe on a CPU?), then 
        this Sequence will break each file into a batch.  """
        
        import numpy as np
        n_files = len(self.file_list)                                                           # get the number of data files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                              # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))                    # the number of batches required to cover every data in the file.  Note that if the batch is set to be larger than the number of data in a file, the ceil function causes this to be 1
        n_batches = n_files * n_batches_per_file                                                # get the number of batches in total to go through all files.          
        return n_batches

    def __getitem__(self, idx):                                             # iterates over the data and returns a complete batch, index is a number upto the number of batches set by __len__, with each number being used once but in a random order.  
        
        import numpy as np
        # repeat of __len__ to get info about batch sizes etc, probably a better way to do this, but tricky as also need things like n_batches_per_file.  
        n_files = len(self.file_list)                                                      # get the number of data files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                         # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))               # the number of batches required to cover every data in the file.  
        n_batches = n_files * n_batches_per_file
        
        # deal with files and batches (convert idx to a file number and batch number).  
        n_file, n_batch = divmod(idx, n_batches_per_file)                                   # idx tells us which batch, but that needs mapping to a file, and to which batch in that file.  
        data = np.load(self.file_list[n_file])                                              # load the correct numpy file.  
        X = data['X']                                                                       # extract X from this (which is needed for the vae)
        Y_class = data['Y_class']                                                           # 
        Y_loc = data['Y_loc']                                                               # 
        
        # Open the correct file and batch, which depends on if it's the last batch for a file (as care needed to make sure it's the same size as all batches).  
        if n_batch == (n_batches_per_file - 1):                                             # the last batch may not have enough data for it, so is more complex to prepare
            X_unused = np.copy(X[n_batch * self.batch_size :, ])                            # get all the data not used in  batches so far, and make part of a batch with it
            Y_class_unused = np.copy(Y_class[n_batch * self.batch_size :, ])                # ditto class labels
            Y_loc_unused = np.copy(Y_loc[n_batch * self.batch_size :, ])                    # dito location labels
            n_required = self.batch_size - X_unused.shape[0]                                # find out how short of a batch we are.  
            extra_data_args = np.arange(n_batch * self.batch_size)                          # get the index of all the data that has been used in the previous batches.  
            np.random.shuffle(extra_data_args)                                              # shuffle it
            X_repeated = np.copy(X[extra_data_args[:n_required]], )                         # and make part of a batch with data that the network has already seen this epoch (ie repeated)
            Y_class_repeated = np.copy(Y_class[extra_data_args[:n_required]], )             # ditto class labels
            Y_loc_repeated = np.copy(Y_loc[extra_data_args[:n_required]], )                 # dito location labels
            
            X_batch = np.concatenate((X_unused, X_repeated), axis = 0)                         # merge the data that make an incomplete batch with some of the repeated data to make the final batch the right size.  
            Y_class_batch = np.concatenate((Y_class_unused, Y_class_repeated), axis = 0)       # ditto class labels
            Y_loc_batch = np.concatenate((Y_loc_unused, Y_loc_repeated), axis = 0)             # dito location labels
    
        else:
            X_batch = X[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]                      # or a batch is just a part of the whole X for that file.  
            Y_class_batch = Y_class[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]          # ditto for class labels
            Y_loc_batch = Y_loc[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]              # and location labels
            
        return X_batch, [Y_class_batch, Y_loc_batch]                                                   # return a batch, created in either of the two ways.  Note that they Ys must be collected into a list in the same was as if you're doing .fit with an array rather than a sequence


#%%
class CustomSaver(keras.callbacks.Callback):
    
    """Ideally model_output_dir could be set as a class variable in __init__, but I haven't worked out how to modify __init__
    and all the methods that it calls, without just copying the whole thing here.  """     
    
    def on_epoch_end(self, epoch, logs={}):                                                                 # overwrite the on_epoch_end default metho in the callback class.  
        self.model.save(str(self.model_output_dir / f"model_epoch_{epoch:03d}"))                  # model_output_dir will have to have been set.  


#%%
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.combined_loss = []                                             # for the training losses, one to be added each batch
        self.class_loss = []
        self.class_accuracy = []
        self.loc_loss = []
        
        self.combined_loss_val = []                                         # the list for the validation losses, one to be added each epoch
        self.class_loss_val = []
        self.class_accuracy_val = []
        self.loc_loss_val = []

    def on_batch_end(self, batch, logs={}):                                                         # every batch, record the loss/accuracy
        # self.combined_loss.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))
        self.combined_loss.append(logs.get('loss'))
        self.class_loss.append(logs.get('class_dense3_loss'))
        self.class_accuracy.append(logs.get('class_dense3_accuracy'))
        self.loc_loss.append(logs.get('loc_dense6_loss'))
        
    def on_epoch_end(self, batch, logs={}):                                                         # every epoch, record the validation loss/accuracy
        self.combined_loss_val.append(logs.get('val_loss'))                                         # the list for the validation losse
        self.class_loss_val.append(logs.get('val_class_dense3_loss'))
        self.class_accuracy_val.append(logs.get('val_class_dense3_accuracy'))
        self.loc_loss_val.append(logs.get('val_loc_dense6_loss'))
        
        
#%%



def plot_vudlnet21_training(vgg16_loss_recorder, n_epochs):
    """Plot the training hitory for the two headed model when loss is recorded for each batch (and for each epoch for the validation data)
    Inputs:
        vgg16_loss_recorder | keras LossHistory, containing lists of the loses (and some accuracie)
        n_epochs | int | number of epochs model was trained for.  
    
    Returns:
        Figure
        
    History:
        2021_07_20 | MEG | Written
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    n_losses_train = len(vgg16_loss_recorder.combined_loss)                             # number of losses in total (ie n_batche * n_epochs)
    n_batches_per_epoch = int(n_losses_train / n_epochs)                                # number of loss values per epochs
    all_epochs = np.arange(0, n_losses_train)                                           # x values for plots every batch, n_files * n_epochs
    validation_epochs = np.arange(-1, n_losses_train, n_batches_per_epoch)[1:]          # x values for plots every epoch, as only do the validation data once an epoch.  First one is negative and not needed
    
    # create the axes and hide those that are not required.      
    f, axes = plt.subplots(3,2, figsize = (16,8))
    for ax in [axes[0,1], axes[2,1]]:
        ax.set_visible(False)
       
    # 1: plot the training loss/accuracy
    axes[0,0].scatter(all_epochs, vgg16_loss_recorder.combined_loss)                                   
    axes[1,0].scatter(all_epochs, vgg16_loss_recorder.class_loss)                                   
    axes[1,1].scatter(all_epochs, vgg16_loss_recorder.class_accuracy)  
    axes[2,0].scatter(all_epochs, vgg16_loss_recorder.loc_loss)  
    
    # 2 : plot the validation loss/accuracy
    #import pdb; pdb.set_trace()
    axes[0,0].scatter(validation_epochs, vgg16_loss_recorder.combined_loss_val)                                   
    axes[1,0].scatter(validation_epochs, vgg16_loss_recorder.class_loss_val)                                   
    axes[1,1].scatter(validation_epochs, vgg16_loss_recorder.class_accuracy_val)  
    axes[2,0].scatter(validation_epochs, vgg16_loss_recorder.loc_loss_val)  
    
    
    # 3 formating/labels etc.  
    axes[0,0].set_title('Loss')
    axes[1,1].set_title('Accuracy')
    axes[2,0].set_xlabel('Epoch #')
    axes[1,1].set_xlabel('Epoch #')
    axes[0,0].set_ylabel('Combined')
    axes[1,0].set_ylabel('Classification')
    axes[2,0].set_ylabel('Location')
    axes[1,1].set_ylim([0, 1])
    
    for axe in np.ravel(axes):
        axe.grid(True, alpha = 0.2, which = 'both')
        axe.set_xlim([0, n_losses_train])
        axe.set_xticks(np.arange(0, n_losses_train, n_batches_per_epoch))                 # change so a tick only after each epoch (and not each file)
        axe.set_xticklabels(np.arange(0, n_epochs))
    for axe in np.ravel(axes[0:1,:]):
        axe.set_xticklabels([])
    
    f.tight_layout()