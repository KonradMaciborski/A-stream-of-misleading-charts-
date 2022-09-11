import multiprocessing as mp
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import albumentations
import numpy as np
import torch
import torch.nn as nn
from albumentations import pytorch
from skorch import NeuralNetClassifier
from skorch.callbacks import Freezer, EarlyStopping
from skorch.callbacks import LRScheduler, Checkpoint
from skorch.dataset import ValidSplit
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from Graphs import Graphs
from DenseNet161 import DenseNet161

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

GPU_NUM = '4'
MODEL_NAME = 'is_a_chart_densenet161_05.pkl'
MAP_NAME = 'is_a_chart_densenet161_map_05.png'

IMG_SIZE = 224
MAX_EPOCHS = 128
DATASET_PATH = '../Is a chart'
SEED = random.randint(2, 2048)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_datasets(dataset_dir, dataset_files):
    data_transforms = albumentations.Compose([
        albumentations.Resize(IMG_SIZE, IMG_SIZE),
        albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        pytorch.transforms.ToTensorV2()
    ])

    dataset = Graphs(dataset_dir, dataset_files, transform=data_transforms)

    test_size = int(len(dataset) * 0.24)
    train_set, test_set = random_split(dataset,
                                       (len(dataset_files) - test_size, test_size))
    print(f'Train dataset length: {len(train_set)}')
    print(f'Testset dataset length: {len(test_set)}')

    return train_set, test_set


def main():
    print("GPU Available: " + str(torch.cuda.is_available()))
    print("Using GPU no." + GPU_NUM)
    print('seed value = ' + str(SEED))

    set_seed(SEED)

    dataset_dir = DATASET_PATH
    dataset_files = os.listdir(dataset_dir)
    random.shuffle(dataset_files)
    print("Dataset size = " + str(len(dataset_files)))

    class_names = ['chart', 'just_image']
    batch_size = 128
    num_workers = mp.cpu_count()
    n_classes = len(class_names)

    train_set, test_set = prepare_datasets(dataset_dir, dataset_files)

    # create dataloaders for loading data in batches=128
    trainloader = DataLoader(train_set, batch_size=batch_size,
                             pin_memory=True, num_workers=num_workers, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_size,
                            pin_memory=True, num_workers=num_workers)

    train_images, train_labels = map(list, zip(*[(X.size(), y)
                                                 for X, y in iter(train_set)]))
    test_images, test_labels = map(list, zip(*[(X.size(), y)
                                               for X, y in iter(test_set)]))

    _, train_counts = np.unique(train_labels, return_counts=True)
    _, test_counts = np.unique(test_labels, return_counts=True)

    # callback functions for models
    # DenseNet169
    # callback for Reduce on Plateau scheduler
    lr_scheduler = LRScheduler(policy='ReduceLROnPlateau',
                               factor=0.5, patience=1)
    # callback for saving the best on validation accuracy model
    checkpoint = Checkpoint(f_params='best_model_densenet169.pkl',
                            monitor='valid_acc_best')
    # callback for freezing all layer of the model except the last layer
    freezer = Freezer(lambda x: not x.startswith('model.classifier'))
    # callback for early stopping
    early_stopping = EarlyStopping(patience=8)

    densenet = NeuralNetClassifier(
        module=DenseNet161,
        module__output_features=n_classes,
        criterion=nn.CrossEntropyLoss,
        batch_size=batch_size,
        max_epochs=MAX_EPOCHS,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=1e-6,
        iterator_train__shuffle=True,
        # load in parallel
        iterator_train__num_workers=num_workers,
        # stratified kfold split of loaded dataset
        train_split=ValidSplit(cv=5, stratified=True),
        # callbacks declared earlier
        callbacks=[lr_scheduler, checkpoint, freezer, early_stopping],
        # use GPU or CPU
        device=("cuda:" + GPU_NUM) if torch.cuda.is_available() else "cpu"
    )

    print('training starting')

    densenet.fit(train_set, y=np.array(train_labels))
    print('Accuracy: {:.5f}%'.format(densenet.score(test_set, test_labels) * 100))

    densenet.save_params(f_params=('../Models/' + MODEL_NAME))

    pred_classes = np.array([])
    true_classes = np.array([])
    for images, labels in iter(testloader):
        pred_classes = np.append(pred_classes, densenet.predict(images))
        true_classes = np.append(true_classes, labels)
    print('DenseNet161 prediction done!')

    cf_matrix = confusion_matrix(true_classes, pred_classes)
    df_cm = pd.DataFrame(cf_matrix.transpose(), index=[i for i in class_names], columns=[i for i in class_names])
    plt.figure(figsize=(16, 8))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../Models/' + MAP_NAME)


if __name__ == '__main__':
    main()
