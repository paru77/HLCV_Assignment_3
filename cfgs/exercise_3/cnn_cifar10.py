from functools import partial
import pickle
import torch
import torch.nn as nn
from copy import deepcopy

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.cnn_trainer import CNNTrainer
from src.models.cnn.model import ConvNet
from src.models.cnn.metric import TopKAccuracy

q1_experiment = dict(
    name = 'CIFAR10_CNN_1Q',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.Identity,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 10,
        eval_period = 1,
        save_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period = 10,
        monitor = "off",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)

#########  TODO #####################################################
#  You would need to create the following config dictionaries       #
#  to use them for different parts of Q2 and Q3.                    #
#  Feel free to define more config files and dictionaries if needed.#
#  But make sure you have a separate config for every question so   #
#  that we can use them for grading the assignment.                 #
#####################################################################
q2a_normalization_experiment = dict(
    name='CIFAR10_CNN_BN2Qa',
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_prob=0.0,
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2",
        transform_preset='CIFAR10',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=10,
        eval_period=1,
        save_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period=10,
        monitor="off",
        early_stop=0,
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)

q2b_experiment1 = dict(
    name = 'CIFAR10_CNN_best_model2b',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.Identity,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)

q2b_experiment2 = dict(
    name = 'CIFAR10_CNN_best_modelBN2b',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)



q2c_earlystop_experiment1 = dict(
    name = 'CIFAR10_CNN_EarlyStopQ1_2c',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.Identity,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 4,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)

q2c_earlystop_experiment2 = dict(
    name = 'CIFAR10_CNN_Q2_C_Q2BN_2c',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 50,
        eval_period = 1,
        save_dir = "/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 4,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)


q3a_aug1_experiment = dict(
    name='CIFAR10_CNN_Q3A_AUG1',
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_prob=0.0,
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2",
        transform_preset='CIFAR10_WithFlip',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=30,
        eval_period=1,
        save_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period=10,
        monitor="max eval_top1",
        early_stop=0,  
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)

q3a_aug2_experiment = dict(
    name='CIFAR10_CNN_Q3A_AUG2',
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_prob=0.0,
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2",
        transform_preset='CIFAR10_WithRotation',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=30,
        eval_period=1,
        save_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period=10,
        monitor="max eval_top1",
        early_stop=0,  
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)

q3a_aug3_experiment = dict(
    name='CIFAR10_CNN_Q3A_AUG3',
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_prob=0.0,
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2",
        transform_preset='CIFAR10_WithColorJitter',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=30,
        eval_period=1,
        save_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period=10,
        monitor="max eval_top1",
        early_stop=0,  
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)

q3a_aug4_experiment = dict(
    name='CIFAR10_CNN_Q3A_AUG4',
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_prob=0.0,
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2",
        transform_preset='CIFAR10_WithAllAug',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=30,
        eval_period=1,
        save_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period=10,
        monitor="max eval_top1",
        early_stop=0,  
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)

dropout_experiment_template = dict(
    name='CIFAR10_CNN_Dropout_Experiment',
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_prob=None,  
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/data/exercise-2",  
        transform_preset='CIFAR10', 
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=30,  
        eval_period=1,
        save_dir="/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Saved",
        save_period=10,
        monitor="max eval_top1",
        early_stop=0, 
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)


dropout_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
dropout_experiments = {}

for p in dropout_probs:
    config = deepcopy(dropout_experiment_template)
    config['name'] += f'_dropout_{p}'
    config['model_args']['drop_prob'] = p
    dropout_experiments[f'dropout_{p}'] = config


import pickle

with open("/content/drive/MyDrive/Colab Notebooks/HLCV Assignment_3/Dropout/dropout_experiments.pkl", "wb") as f:
    pickle.dump(dropout_experiments, f)
  