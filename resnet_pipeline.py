import shutil

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_path = 'APTOS.npz'

# Parsing data and splitting into train test
parse_data_to_folders(dataset_path=dataset_path)
make_train_test_split('total_data', [0.7, 0.3])
budget_percentage = [5, 10, 15, 20, 25, 30]
exp_name = 'exp3'

epochs = 100
optimizer = 'Adam'
batch_size = 16
weight_decay = 0.0005
learning_rate = 0.01


def train_resnet18(train_folder_path, test_folder_path, epochs, batch_size, learning_rate, weight_decay, optimizer_str):
    try:
        os.mkdir('resnet18_models')
    except FileExistsError:
        pass

    # try:
    #     os.mkdir('outputs')
    # except FileExistsError:
    #     pass

    # Set seed
    seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # np.random.seed(seed)
    # random.seed(seed)
    torch.random.manual_seed(42)
    num_classes = 2

    train_loader, valid_loader = get_data(train_folder_path, test_folder_path, batch_size=batch_size)

    # Define model based on the argument parser string.
    # if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes).to(device)
    plot_name = 'resnet_scratch'

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    if optimizer_str == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_str == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_str == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print('Optimizer not defined ...')
        exit()

    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.

    max_val_acc = 0

    start_time = time.time()

    # Code will analyse last es_epochs to decide on early stopping of model
    es_epochs = 20

    # If max change in valid acc of last es_epochs fall below es_delta then training will stop
    es_delta = 0.01

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc, model, per_cls_train_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_cls=num_classes
        )
        valid_epoch_loss, valid_epoch_acc, model, per_cls_test_acc = validate(
            model,
            valid_loader,
            criterion,
            device,
            num_cls=num_classes
        )

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        if valid_epoch_acc > max_val_acc:
            print('Saving best.pth to working directory ... ')
            # print(f'Per class test acc for this model is - {per_cls_test_acc}')
            torch.save(model.state_dict(), 'resnet18_models/best.pth')
            max_val_acc = valid_epoch_acc
        else:
            print('Saving last.pth to working directory ... ')
            torch.save(model.state_dict(), 'resnet18_models/last.pth')

        # Early Stopping Code
        # if epoch < es_epochs:
        #     pass
        # else:
        #     # std_of_last_es_epochs = np.std(valid_acc[::-es_epochs])
        #     max_delta_of_last_es_epochs = np.round(max(valid_acc[::-es_epochs]) - min(valid_acc[::-es_epochs]), 2)
        #     print(f'Max delta of validation acc for last {es_epochs} epochs - {max_delta_of_last_es_epochs}')
        #
        #     if max_delta_of_last_es_epochs < es_delta and train_acc[-1] > 96:
        #         print(f'\n\nNo significant improvement in model found for last {es_epochs}')
        #         print('Early stopping model training ... \n\n')
        #         break
        #     else:
        #         pass

        # Change round off parameters to see exact accuracies (e.g. :.3f round off to 3 decimal places)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.2f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.2f}")
        # Printing per class accuracy after round off to 1 decimal place
        print(f'Per class test accuracy - {np.round_(per_cls_test_acc, decimals=2)}')
        print('-' * 50, '\n')

    end_time = time.time()

    print('\n\nSummary')
    print(f'Max training accuracy - {np.round(max(train_acc), 2)}')
    print(f'Max validation accuracy - {np.round(max(valid_acc), 2)}')
    print(f'Time required for training model - {np.round((end_time - start_time) / 60, 2)} \n\n')

    # Save the loss and accuracy plots.
    # save_plots(
    #     train_acc,
    #     valid_acc,
    #     train_loss,
    #     valid_loss,
    #     name=plot_name
    # )
    # print('TRAINING COMPLETE')

    return np.max(train_acc), np.max(valid_acc)

budget_array = []
train_acc_array = []
test_acc_array = []
for i in range(len(budget_percentage)):

    try:
        shutil.rmtree('to_train')
    except FileNotFoundError:
        pass

    percent_random_select(folder_to_split='train_data', percentage_to_select=budget_percentage[i], folder_save_path='to_train')

    temp_train_acc, temp_test_acc = train_resnet18(train_folder_path='to_train', test_folder_path='test_data',
                                                   epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                                   weight_decay=weight_decay, optimizer_str=optimizer)

    results = pd.DataFrame()

    budget_array.append(budget_percentage[i])
    train_acc_array.append(temp_train_acc)
    test_acc_array.append(temp_test_acc)

    results['budget'] = budget_array
    results['train_acc'] = train_acc_array
    results['test_acc'] = test_acc_array

    results.to_excel(f'{exp_name}_{dataset_path.split(".")[0]}_resnet18.xlsx', index=False)

# Deleting folders and files to assist next experiment setup
files_to_delete = ['lake', 'test_data', 'train_data', 'total_data', 'to_train', 'resnet18_models', '__pycache__']
for i in range(len(files_to_delete)):
    try:
        shutil.rmtree(files_to_delete[i])
    except FileNotFoundError:
        pass

print('Done')
