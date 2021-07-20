#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:27:44 2021

@author: matthew
"""

  


#%%

def plot_batchwise_history(history_dict, outpath = None):
    """Given a dict of loss for each batch of training data and validation loss for the end of each epoch, plot them.   
    Inputs:
        history_dict | dict | usually history.history 
        outpath | string | path to file, including .png extension.  
    Returns
        figure
    History: 
        2021_05_21 | MEG | Written
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    #0: get the number of batches (and the xvales for plotting things)
    n_epochs = len(history_dict['val_loss'])
    n_batches =  int(len(history_dict['loss']) / n_epochs)                                  # number of batches of training data per epoch.  
    
    xvals_batches = np.arange(0, len(history_dict['loss']))                                         # x values for plotting training data, as there's (n_files * n_epochs) of these values
    xvals_epochs = np.arange(-1, len(history_dict['loss']), n_batches)[1:]                    # x values for plotting validation data, n_epochs as only do the validation data once an epoch
        
    #1: Make the figure
    f, ax = plt.subplots(1,1, figsize = (10,5))
    ax.plot(xvals_batches, history_dict['loss'], label = 'Loss')
    ax.plot(xvals_epochs, history_dict['val_loss'], label = 'Validation Loss')
    
    # 2: formatting of the figure
    ax.grid(True, alpha = 0.2, which = 'both')

    ax.set_xlim([0, (xvals_epochs[-1]+1)])
    ax.set_ylim(bottom = 0)
    ax.set_xticks(np.arange(0, (n_batches * n_epochs)+1, n_batches))                                    #  change so a tick only after each epoch (and not each file)
    ax.set_xticklabels(np.arange(0,n_epochs+1))                                                  # number ticks
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch n')
    plt.legend(loc = 'upper right')
    
    if outpath != None:
        f.savefig(outpath)

#%%

 
def model_runtime_predictor(keras_model, files_train, files_validate, n_epochs= 1, outdir = None):
    """ Calcaute the time expected to be required to train the model.  Would normally be used 
    to calcaulte the amount of time to request from a scheduler (e.g. on ARC)
    Inputs:
        keras_model | keras model | the model to be trained.  
        files_train | list | a list of paths to the files used for training.  
        files_validate | list |a list of paths to the files used for validation.  
        n_epochs | int | the number of epochs that the model wil be trained for.  
    Returns:
        timing_prediction.txt | file | Contains timing information.  
    History:
        2021_03_19 | MEG | Written
    """
    import time
    import numpy as np
    from pathlib import Path
    

    #0: Open one file and fit the mode with it, and time the whole process    
    tic = time.perf_counter()
    data = np.load(files_train[0])
    X_batch = data['X']
    Y_batch = data['Y']
            
    keras_model.fit(X_batch, Y_batch, batch_size=32, epochs=1, verbose = 1)                      # train it on one file
    toc = time.perf_counter()
    time_delta = toc - tic

#    import pdb; pdb.set_trace()        
    # 1: work out how long it will take to train the whole model.  
    if outdir == None:
        outfile = open("timing_prediction.txt", 'w')
    else:
        outfile = open(Path(f"{outdir}/timing_prediction.txt"), 'w')
        
    _, time_string = sec_to_dhms(time_delta)                                                                            # convert from seconds to useful units 
    outfile.write(f"Opening one file and fitting the model with it took {time_string} (dd:hh:mm:ss) to train.  \n")
        
    time_delta = time_delta * (len(files_train) + len(files_validate))                                                  # the time for one epoch, which is dependent on the number of training files and the number of validation files
    _, time_string = sec_to_dhms(time_delta)                                                                            # convert from seconds to useful units 
    outfile.write(f"One epoch ({len(files_train)} training files and {len(files_validate)} validation files) is predicted to take {time_string} (dd:hh:mm:ss) to train.  \n")
    
    time_delta *=  n_epochs
    _, time_string = sec_to_dhms(time_delta)                                                                            # convert from seconds to useful units 
    outfile.write(f"{n_epochs} epochs are expected to take {time_string} (dd:hh:mm:ss) to train.  \n")
    
    outfile.close()


#%%

def sec_to_dhms(t_seconds):
    """ Given a time in seconds, convert it to dhms format.  Note that by changing "units in seconds" (which is hardcoded below)
    any units could be provided.  e.g. years could be added.  
    
    Inputs:
        t_seconds | int or float |  the number of seconds to convert
    Returns:
        n_units | list of ints | number of days, hours, minutes, seconds, rounded to the nearest whole number.  
        time_string | string | the units as a string separated by colons.  
    History:
        2021_03_23 | MEG | Written.  
    
    """
    import numpy as np
    
    units_in_seconds = [86400, 3600, 60, 1]                         # day, hour, minute, seconds    all, in seconds.  
    n_units = []                                                    # this will store the number of the above units we have.  
    time_string = ''
    remainder_sec = t_seconds                                       # to initiate loop
    for unit_in_seconds in units_in_seconds:                        # loop through each unit
        n_unit = np.floor(remainder_sec / unit_in_seconds)          # calculate how many of them we have (i.e. divide but just keep the integer bit and round down.  )
        time_string = time_string + str(int(n_unit)) + ':'               # also add it to a long string
        n_units.append(int(n_unit))                                 # append to the list of how many units we have
        remainder_sec = remainder_sec % unit_in_seconds             # find the remainder, which will be used by the next unit in the loop.  
    time_string = time_string[:-1]                                   # the last character is a : so delete it 
    return n_units, time_string


#%%

def train_unw_network(model, files, n_epochs, loss_names, X_validate, Y_validate, outdir, 
                      round_dp = 4, figure = False):
    """Train a double headed model using training data stored in separate files.  
    Inputs:
        model | keras model | the model to be trained
        files | list | list of paths and filenames for the files used during training
        n_epochs | int | number of epochs to train for
        loss names | list | names of outputs of losses (e.g. "class_dense3_loss)
        X_validate
        Y_validate
        outdir 
        round_dp | int | number of decimal points to round loss that is printed to terminal to.  
        figure | boolean | if True, a png is produced showing current loss values.  Can cause errors due to X forwarding.  
    Returns
        model | keras model | updated by the fit process
        metrics_loss | r2 array | columns are: total loss/class loss/loc loss /validate total loss/validate class loss/ validate loc loss
        metrics_class | r2 array | columns are class accuracy, validation class accuracy
        
    2019/03/25 | Written.  
    2021/03/08 | MEG | Save the model after each epoch.  A crude approach to early stopping
    """
    import numpy as np
    import keras
    from pathlib import Path
    
    def plot_single_train_validate_loss(metrics_loss, n_files, outdir, n_epoch,
                                        pointsize = 2, spacing = 2, no_display = True):
        """ Create a plot showing training and validation loss when training using data split across multiple files.  
        Inputs:
            metrics_loss | numpy array | (n_files * n_epochs) x 2, becuase there is a loss for each file in every epoch.  First column for training, second for validation (so mostly 0s as only 1 pass per epoch)
            n_files | int | number of files being passed
            out_dir | string or Path | directory to save the figure to
            n_epoch | which epoch number currently up to (i.e epoch 5 of 10)
            pontsize | int | size of points in plot
            spacing | int | every nth epoch is labelled on the x axis.  
            no_display | boolean | If True, use the Agg backend of matplotlib that doesn't need X forwarding to work.  
            
        """
        
        import matplotlib
        if no_display:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        
        all_epochs = np.arange(0, metrics_loss.shape[0])                                         # n_files * n_epochs
        validation_epochs = np.arange(-1, metrics_loss.shape[0], n_files)[1:]                    # n_epochs (as only do the validation data once an epoch)
        
               
        f, ax = plt.subplots(1,1)
        ax.scatter(all_epochs[:n_files*(n_epoch+1)], metrics_loss[:,0][:n_files*(n_epoch+1)], s=pointsize )                          # training loss
        ax.scatter(validation_epochs[:(n_epoch+1)], metrics_loss[validation_epochs,1][:(n_epoch+1)], s=pointsize )                   # validation loss
        
        ax.grid(True, alpha = 0.2, which = 'both')
        ax.set_xlim([0, all_epochs.shape[0]])
        ax.set_ylim(bottom = 0)
        ax.set_xticks(np.arange(0,n_files * n_epochs,spacing * n_files))                 # change so a tick only after each epoch (and not each file)
        ax.set_xticklabels(np.arange(0,n_epochs, spacing))                                  # number ticks
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch n')
        f.savefig(Path(f"{outdir}/epoch_{n_epoch:003}_training_progress.png"), bbox_inches='tight')
        plt.close(f)
    
    n_files_train = len(files)                                                              # get the number of training files
    
    metrics_loss = np.zeros((n_files_train*n_epochs, 2))                                     # total loss/class loss/loc loss /validate total loss/validate class loss/ validate loc loss
    
    for e in range(n_epochs):                                                                        # loop through the number of epochs
        for file_num, file in enumerate(files):                                   # for each epoch, loop through all files once
        
            data = np.load(file)
            X_batch = data['X']
            Y_batch = data['Y']
            
            history_train_temp = model.fit(X_batch, Y_batch, batch_size=32, epochs=1, verbose = 0)                      # train it on one file
            metrics_loss[(e*n_files_train)+file_num, 0] = history_train_temp.history['loss'][0]                        # main loss    
            print(f'Epoch {e}, file {file_num}: Loss = {round(metrics_loss[(e*n_files_train)+file_num, 0], round_dp)} ')

            
        history_validate_temp = model.evaluate(X_validate, Y_validate, batch_size = 32, verbose = 0)                    # predict on validation data
        metrics_loss[(e*n_files_train)+file_num, 1] = history_validate_temp[0]                                          # main loss, validation
        print(f'Epoch {e}, valid.: Loss = {round(metrics_loss[(e*n_files_train)+file_num, 1], round_dp)} ')
        
        print(f"Saving the current model...", end = '')
        model.save(Path(f"{outdir}/epoch_{e:03}_model_weights"))                                                        # save the model after each epoch
        print('Done.  ')
        
        #pdb.set_trace()
        if figure:
            plot_single_train_validate_loss(metrics_loss, len(files), outdir, e)

        
    
    return model, metrics_loss




#%%

def plot_train_validate_loss(metrics_loss, loss_names, n_files, outdir, n_epoch, pointsize = 2, spacing = 2, no_display = True):
    """ Create a plot showing training and validation loss when training using data split across multiple files.  
    Inputs:
        metrics_loss | numpy array | (n_files * n_epochs) x 2, becuase there is a loss for each file in every epoch.  First column for training, second for validation (so mostly 0s as only 1 pass per epoch)
        n_files | int | number of files being passed
        out_dir | string or Path | directory to save the figure to
        n_epoch | which epoch number currently up to (i.e epoch 5 of 10)
        pontsize | int | size of points in plot
        spacing | int | every nth epoch is labelled on the x axis.  
        no_display | boolean | If True, use the Agg backend of matplotlib that doesn't need X forwarding to work.  
    Returns:
        Figure
    History:
        2021_05_XX | MEG | Modified from an existing script.  
        
        
    """
    
    import numpy as np
    import matplotlib
    from pathlib import Path
    
    if no_display:
        matplotlib.use('Agg')                                                               # works on server (e.g. ARC)
    import matplotlib.pyplot as plt
    
    all_epochs = np.arange(0, metrics_loss.shape[0])                                         # x values for plotting training data, as there's (n_files * n_epochs) of these values
    validation_epochs = np.arange(-1, metrics_loss.shape[0], n_files)[1:]                    # x values for plotting validation data, n_epochs as only do the validation data once an epoch
    

    # 0: Check some of the inputs
    if metrics_loss.shape[1] % 2 != 0:
        raise Exception(f"The number of columns of 'metrics_loss' must always be equal as for every loss, there must be a value for training and validation loss.  Exiting.")
    if metrics_loss.shape[1] != 2*(len(loss_names)):
        raise Exception(f"The number of columns of 'metrics_loss' must always be twice the length of 'loss_names', as it contains a training and validation loss for each type of loss.  Exiting.  ")
    n_losses = len(loss_names)
    

    # 1: begin the plot
    f, axes = plt.subplots(1, n_losses)
    axes = np.atleast_1d(axes)                                                                                                      # if there's only one plot, we need it to be at leas 1d to be able to loop though it
    
    for loss_n, ax in enumerate(axes):
        ax.scatter(all_epochs[:n_files*(n_epoch+1)], metrics_loss[:,loss_n][:n_files*(n_epoch+1)], s=pointsize )                          # training loss
        ax.scatter(validation_epochs[:(n_epoch+1)], metrics_loss[validation_epochs,loss_n + n_losses][:(n_epoch+1)], s=pointsize )        # validation loss
        
        # determine if we need to change the y limit to make the graph legible (due to possibly huge loss with first file of first epoch).  
        if n_epoch > 0:                                                                         # dont need to do that for the first epoch
            epoch0_max_loss = metrics_loss[0, loss_n]                                           # assume first loss of epoch is highest
            epoch1_max_loss = metrics_loss[n_files, loss_n]                                     # assume first loss of epoch is highest
            if epoch0_max_loss > 2 * epoch1_max_loss:                       
                ax.set_ylim(top = epoch1_max_loss)
        
        
        # Small formatting steps.  
        ax.grid(True, alpha = 0.2, which = 'both')
        ax.set_xlim([0, all_epochs.shape[0]])
        ax.set_ylim(bottom = 0)
        ax.set_xticks(np.arange(0,n_files * n_epoch,spacing * n_files))                                    #  change so a tick only after each epoch (and not each file)
        ax.set_xticklabels(np.arange(0,n_epoch, spacing))                                                  # number ticks
        ax.set_ylabel(loss_names[loss_n])
        ax.set_xlabel('Epoch n')
        
        
        
    f.savefig(Path(f"{outdir}/epoch_{n_epoch:003}_training_progress.png"), bbox_inches='tight')
    plt.close(f)
