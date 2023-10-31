import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import shutil


def parse_data_to_folders(dataset_path: str) -> None:
    """

    :param dataset_path: Path to dataset (.npz file) to extract. Dataset should have dict with 'train_labels' and 'train_images' as keys
    :return: None
    """

    data = np.load(dataset_path)

    try:
        os.mkdir('total_data')
    except FileExistsError:
        pass

    try:
        os.mkdir('total_data/0')
    except FileExistsError:
        pass

    try:
        os.mkdir('total_data/1')
    except FileExistsError:
        pass

    train_images = data['train_images']
    train_labels = data['train_labels']

    for i in tqdm(range(len(train_labels))):

        img = train_images[i]

        if train_labels[i] == 0:
            plt.imsave(f'total_data/0/0_{i}.jpg', img)
            # plt.imsave(f'lake/0_{i}.jpg', img)
        else:
            plt.imsave(f'total_data/1/1_{i}.jpg', img)
            # plt.imsave(f'lake/1_{i}.jpg', img)

    print('\nDataset creation complete\n')


def make_train_test_split(data_folder_path: str, train_test_split: list):
    assert np.sum(train_test_split) == 1, 'Sum of fractions in train_test_split should be equal to 1'
    classes = os.listdir(data_folder_path)
    num_cls = len(classes)

    try:
        os.mkdir('test_data')
    except FileExistsError:
        pass

    try:
        os.mkdir('train_data')
    except FileExistsError:
        pass

    try:
        os.mkdir('lake')
    except FileExistsError:
        pass

    # Making folders
    for i in range(num_cls):
        try:
            os.mkdir(f'train_data/{classes[i]}')
        except FileExistsError:
            pass

        try:
            os.mkdir(f'test_data/{classes[i]}')
        except FileExistsError:
            pass

    num_imgs_per_class = []
    num_train_per_class = []
    num_test_per_class = []
    for i in range(num_cls):
        num_imgs_per_class.append(len(os.listdir(os.path.join(data_folder_path, classes[i]))))
        num_train_per_class.append(int(num_imgs_per_class[i] * train_test_split[0]))
        num_test_per_class.append(int(num_imgs_per_class[i] - num_train_per_class[i]))

    print(f'Number of images per class - {num_imgs_per_class}')
    print(f'Number of train images per class - {num_train_per_class}')
    print(f'Number of test images per class - {num_test_per_class}')

    print('Copying files ...')
    for i in range(num_cls):
        for j in tqdm(range(num_imgs_per_class[i])):

            images = os.listdir(os.path.join(data_folder_path, classes[i]))

            img_source = os.path.join(data_folder_path, classes[i], images[j])

            if j < num_train_per_class[i]:
                img_destination = os.path.join('train_data', classes[i], images[j])

                # Also simultaneously copying files to lake
                shutil.copy(img_source, os.path.join('lake', images[j]))
            else:
                img_destination = os.path.join('test_data', classes[i], images[j])

            shutil.copy(img_source, img_destination)

    print('\nData split completed ...\n')


def percent_random_select(folder_to_split: str, percentage_to_select: float, folder_save_path: str) -> None:
    num_cls = len(os.listdir(folder_to_split))
    class_names = os.listdir(folder_to_split)

    try:
        os.mkdir(folder_save_path)
    except FileExistsError:
        print('Warning - Folder already exists')
        pass

    for i in range(num_cls):
        try:
            os.mkdir(os.path.join(folder_save_path, class_names[i]))
        except FileExistsError:
            pass

        imgs_cls = os.listdir(os.path.join(folder_to_split, class_names[i]))
        random.shuffle(imgs_cls)

        num_imgs_cls = len(imgs_cls)
        select_number = int(percentage_to_select * num_imgs_cls / 100)

        print(f'Number of images in class {class_names[i]} - {select_number}')

        for j in tqdm(range(select_number)):
            shutil.copy(os.path.join(folder_to_split, class_names[i], imgs_cls[j]),
                        os.path.join(folder_save_path, class_names[i], imgs_cls[j]))

    return
