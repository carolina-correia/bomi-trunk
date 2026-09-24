#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from emg_regression.utils.data_processing import Data_processing, combine_datasets
from cfg.tools import load_yaml, get_subj_day_folder

def process_data(data_path,save_path):
    # Load params
    params     = load_yaml("cfg/params.yaml")
    emg_params = load_yaml("cfg/emg.yaml")

    # print(data_path, save_path)

    data_processing = Data_processing(data_path, emg_params,
                                    fs_recording=params['record']['freq'],
                                    degrees=False, 
                                    downsample_factor=1, 
                                    debug=1)

    data = data_processing.run()

    # # data_processing.remove_trials(trial_ids_to_remove=[45]) # trial i+1

    # Remove last seconds of training
    crop_duration = 7 # seconds -1s before start task and 6 seconds of movement
    nb_samples_train = int(crop_duration * data_processing.fs_features)
    data = data[:,:nb_samples_train,:] # get just first 5 seconds

    # all trials starting from 0 time
    data[:,:,0] = data[:,:,0] - data[:,0,0][:,np.newaxis]

    # Save dataset
    data_processing.dataset = data.transpose(1,0,2)
    data_processing.save(save_path)


if __name__ == "__main__":

    # Load params
    params     = load_yaml("cfg/params.yaml")
    subj_day_folder = get_subj_day_folder(params['load']['subj'],params['load']['date'])
    record_path = f"{subj_day_folder}{params['interface']['task']}/{params['load']['modality']}"
    train_files = load_yaml(f"{subj_day_folder}files.yaml")['training']
    # print(data_path, save_path)

    for file_name in train_files:
        data_path = f"{record_path}/data_{file_name}.pkl" #raw data
        save_path = f"{record_path}/imuemg_{file_name}"   #processed data    
        process_data(data_path, save_path)

    # Combine processed data
    alldata = combine_datasets(record_path,files=train_files)
    # alldata = alldata[100:-100,:,:]

    # remove baseline EMG:
    # alldata[:,:,5:9] = alldata[:,:,5:9] - alldata[:bl_samples,:,5:9].mean(0)[np.newaxis]

    # check the data
    a = 0
    plt.subplots(figsize=(10,7))
    for k in range(alldata.shape[1]):
        plt.scatter(alldata[a:,k,1],alldata[a:,k,2],s=2)
    plt.show()

    print("Dataset points:", alldata.shape[0]*alldata.shape[1])

    # check emg data
    bl_samples = 100
    nb_traj = alldata.shape[1]
    num_cols = min(nb_traj, 8)  # Maximum of 8 rows
    num_rows = (nb_traj+num_cols-1) // num_cols  # Calculate the number of columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14,8))  # Adjust figsize as neede
    for i,ax in enumerate(axes.flatten()):
        if i < nb_traj:
            ax.plot(alldata[:,i,0],alldata[:,i,5:9])
            # ax.plot(alldata[:,i,0],alldata[:,i,5:9] - alldata[:bl_samples,i,5:9].mean(0) )
    plt.tight_layout()
    plt.show()
