#!/usr/bin/env python3
import os, sys, yaml
import datetime, pickle
import numpy as np


def set_paths(subj,print_):
    """ create subjects' today's folder and task folder"""
    path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
    subj_dir = path + "data/subjects/" + subj
    date  = datetime.datetime.now().strftime("%d_%m_%Y")
    subj_day_folder = f"{subj_dir}/{date}"
    other_folders = ['training/imu','testing/imu','model','interface','figs']

    if not os.path.isdir(subj_dir): 
        os.makedirs(subj_dir)
    if not os.path.isdir(subj_day_folder): 
        os.makedirs(subj_day_folder, exist_ok=True)
        if print_:
            print('='*20,'\nSubject dir:',subj_dir)
            print("Folders created:", subj_day_folder)
            print('='*20)    
    # create the other folders
    for folder in other_folders:
        folder_path = os.path.join(subj_day_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    files_path = f'{subj_day_folder}/files.yaml'
    if not os.path.exists(files_path):
        with open(files_path, 'w') as yaml_file:
            yaml.dump({'training': [],'testing': []}, yaml_file, sort_keys=False)
    return subj_day_folder


def get_subj_day_folder(subj, date):
    path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')

    if date is None:
        date_  = datetime.datetime.now().strftime("_%d_%m_%Y")
    else:
        date_ = date
    subj_dir = path + "data/subjects/" + subj + "/"
    subj_day_folder = subj_dir + date_ + "/"
    return subj_day_folder

# def get_data_path(path,subj,date,control_modality,filename):
#     subj_task_folder = path + "data/subjects/" + subj + "/" + date + "/" + control_modality+ "/"
#     return subj_task_folder + 'data_' + filename +'.pkl'


def load_yaml(file_path):
    with open(file_path, "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.SafeLoader)
