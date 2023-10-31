import shutil
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
from resnet18 import ResNet, BasicBlock
from training_utils import train, validate
from utils import get_data
import time
import torch.nn as nn
import pandas as pd
from parse_data_and_split import parse_data_to_folders, make_train_test_split, percent_random_select
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_path = 'APTOS.npz'

# Parsing data and splitting into train test
parse_data_to_folders(dataset_path=dataset_path)
make_train_test_split('total_data', [0.7, 0.3])
budget_percentage = [5, 10, 15, 20, 25, 30]
exp_name = 'exp2'

epochs = 100
optimizer = 'Adam'
batch_size = 16
weight_decay = 0.0005
learning_rate = 0.01


def create_npz_from_train_and_test(train_folder_path: str, test_folder_path: str, save_npz_path: str) -> None:

    save_dict = {}

    train_cls = os.listdir(train_folder_path)
    num_train_cls = len(train_cls)

    train_images = []
    train_labels = []
    for i in range(num_train_cls):

        imgs = os.listdir(os.path.join(train_folder_path, train_cls[i]))
        num_imgs = len(imgs)

        for j in range(num_imgs):
            train_images.append(plt.imread(os.path.join(train_folder_path, train_cls[i], imgs[j])))
            train_labels.append(i)

        # Combine the lists into a single list of tuples
        combined_lists = list(zip(train_images, train_labels))

        # Shuffle the combined list
        random.shuffle(combined_lists)

        # Unzip the shuffled list back into separate lists
        train_images, train_labels = zip(*combined_lists)

        # Convert the shuffled tuples back to lists
        train_images = list(train_images)
        train_labels = list(train_labels)

    save_dict['train_images'] = train_images
    print(f'Number of train images - {len(train_images)}')
    save_dict['train_labels'] = train_labels
    print(f'Number of train labels - {len(train_labels)}')


    test_cls = os.listdir(test_folder_path)
    num_test_cls = len(test_cls)

    test_images = []
    test_labels = []
    for i in range(num_test_cls):
        # print(i)

        imgs = os.listdir(os.path.join(test_folder_path, test_cls[i]))
        num_imgs = len(imgs)

        for j in range(num_imgs):
            test_images.append(plt.imread(os.path.join(test_folder_path, test_cls[i], imgs[j])))
            test_labels.append(i)

        # Combine the lists into a single list of tuples
        combined_lists = list(zip(test_images, test_labels))

        # Shuffle the combined list
        random.shuffle(combined_lists)

        # Unzip the shuffled list back into separate lists
        test_images, test_labels = zip(*combined_lists)

        # Convert the shuffled tuples back to lists
        test_images = list(test_images)
        test_labels = list(test_labels)

    save_dict['test_images'] = test_images
    print(f'Number of test images - {len(test_images)}')
    save_dict['test_labels'] = test_labels
    print(f'Number of test images - {len(test_images)}')

    # Save the dictionary to an NPZ file
    np.savez(save_npz_path, **save_dict)

    return


budget_array = []
train_acc_array = []
test_acc_array = []
for i in range(len(budget_percentage)):

    try:
        shutil.rmtree('to_train')
    except FileNotFoundError:
        pass

    percent_random_select(folder_to_split='train_data', percentage_to_select=budget_percentage[i],
                          folder_save_path='to_train')

    create_npz_from_train_and_test(train_folder_path='to_train', test_folder_path='test_data',
                                   save_npz_path=f'data/{dataset_path.split(".")[0]}_{budget_percentage[i]}.npz')

    # results = pd.DataFrame()

    # budget_array.append(budget_percentage[i])
    # train_acc_array.append(temp_train_acc)
    # test_acc_array.append(temp_test_acc)

    # results['budget'] = budget_array
    # results['train_acc'] = train_acc_array
    # results['test_acc'] = test_acc_array
    #
    # results.to_excel(f'{exp_name}_{dataset_path.split(".")[0]}_resnet18.xlsx', index=False)


# Deleting folders and files to assist next experiment setup
files_to_delete = ['lake', 'test_data', 'train_data', 'total_data', 'to_train', 'resnet18_models', '__pycache__']
for i in range(len(files_to_delete)):
    try:
        shutil.rmtree(files_to_delete[i])
    except FileNotFoundError:
        pass

print('Done')


