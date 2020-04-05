#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from uuunet.run.default_configuration import get_default_configuration
from uuunet.paths import default_plans_identifier
from uuunet.paths import network_training_output_dir
from uuunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from uuunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from uuunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes

from uuunet.MAGIC import *
from uuunet.resnet import *
import torch

import os
import pickle
import time
import shutil
import tensorboardX

def modify_plans_2d():
    ## MR
    path = "preprocessed/Task00_CHAOSMR/nnUNetPlans_plans_2D.pkl"
    with open(path, "rb") as f:
        plan = pickle.load(f)
    plan["plans_per_stage"][0]['patch_size'] = [256, 256]
    plan["plans_per_stage"][0]['batch_size'] = 16
    with open(path, "wb") as f:
        pickle.dump(plan, f)
    ## CT
    path = "preprocessed/Task01_CHAOSCT/nnUNetPlans_plans_2D.pkl"
    with open(path, "rb") as f:
        plan = pickle.load(f)
    plan["plans_per_stage"][0]['patch_size'] = [512, 512]
    plan["plans_per_stage"][0]['batch_size'] = 4
    with open(path, "wb") as f:
        pickle.dump(plan, f)


def modify_plans_3d():
    ## MR
    path = "preprocessed/Task00_CHAOSMR/nnUNetPlans_plans_3D.pkl"
    with open(path, "rb") as f:
        plan = pickle.load(f)
    plan["plans_per_stage"][0]['patch_size'] = [32, 256, 256]
    plan["plans_per_stage"][0]['batch_size'] = 2
    with open(path, "wb") as f:
        pickle.dump(plan, f)
    ## CT
    path = "preprocessed/Task01_CHAOSCT/nnUNetPlans_plans_3D.pkl"
    with open(path, "rb") as f:
        plan = pickle.load(f)
    plan["plans_per_stage"][0]['patch_size'] = [64, 512, 512]
    plan["plans_per_stage"][0]['batch_size'] = 2
    plan["plans_per_stage"][1]['patch_size'] = [64, 512, 512]
    plan["plans_per_stage"][1]['batch_size'] = 2
    with open(path, "wb") as f:
        pickle.dump(plan, f)

def backup(logdir):
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(logdir, "backup", time_stamp)
    os.makedirs(path, exist_ok=True)
    files = ['run2.sh', 'uuunet/MAGIC.py', 'uuunet/run/run_training.py', 'uuunet/paths.py', 'uuunet/resnet.py']
    for fro in files:
        to = os.path.join(path, os.path.basename(fro))
        shutil.copy(fro, to)
        print(f"[BACKUP] {fro} --> {to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")

    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-t", "--task", nargs='+', required=True) # haw needs this to be modified.

    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier", default=default_plans_identifier, required=False)
    parser.add_argument("-u", "--unpack_data", help="Leave it as 1, development only", required=False, default=1,
                        type=int)
    parser.add_argument("--ndet", help="Per default training is deterministic, "
                                                   "nondeterministic allows cudnn.benchmark which will can give up to "
                                                   "20%% performance. Set this to do nondeterministic training",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the vlaidation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    
    args = parser.parse_args()

    task0, task1 = args.task # haw

    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p

    unpack = args.unpack_data
    deterministic = not args.ndet


    if unpack == 0:
        unpack = False
    elif unpack == 1:
        unpack = True
    else:
        raise ValueError("Unexpected value for -u/--unpack_data: %s. Use 1 or 0." % str(unpack))

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # backup 
    backup(network_training_output_dir)

    # two dataset
    plans_file0, output_folder_name0, dataset_directory0, batch_dice0, stage0, trainer_class0 = get_default_configuration(network, task0, network_trainer, plans_identifier)
    plans_file1, output_folder_name1, dataset_directory1, batch_dice1, stage1, trainer_class1 = get_default_configuration(network, task1, network_trainer, plans_identifier)

    # tweak patch_size, batch_size
    modify_plans_2d()

    # I will use two trainers, but sharing the same model
    
    trainer0 = trainer_class0(plans_file0, fold, output_folder=output_folder_name0, dataset_directory=dataset_directory0, batch_dice=batch_dice0, stage=stage0, unpack_data=unpack, deterministic=deterministic, fp16=False)
    trainer1 = trainer_class1(plans_file1, fold, output_folder=output_folder_name1, dataset_directory=dataset_directory1, batch_dice=batch_dice1, stage=stage1, unpack_data=unpack, deterministic=deterministic, fp16=False)

    trainer0.initialize(not validation_only)
    trainer1.initialize(not validation_only)

    ##################################
    #  the magic to merge two trainer.
    

    network = XNet()
    #network = YNet()
    #network = DANet()
    network.cuda()
    
    # how about seperated optimizer?
    optimizer0 = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_scheduler0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer0, mode='min', factor=0.2, patience=5)

    optimizer1 = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.2, patience=5)
    
    writer = tensorboardX.SummaryWriter(os.path.join(network_training_output_dir, "run")) # path?

    trainer0.network = network
    trainer0.optimizer = optimizer0
    trainer0.lr_scheduler = lr_scheduler0
    trainer0.writer = writer

    trainer1.network = network
    trainer1.optimizer = optimizer1
    trainer1.lr_scheduler = lr_scheduler1
    trainer1.writer = writer
    
    #######################################
    # how to train it ?
    # * first MR full epochs, then CT full epochs
    # * one epoch MR, one epoch CT
    # * one step MR, one step CT

    if not validation_only:
        if args.continue_training:
            trainer0.load_latest_checkpoint() 
            trainer1.load_latest_checkpoint() 
        ### train it
        #run_training_B(trainer0, trainer1)
        run_training_C(trainer0, trainer1)
        #run_training_A(trainer0, 0)
        #run_training_A(trainer1, 1)
    else:
        trainer0.load_latest_checkpoint(train=False)
        trainer1.load_latest_checkpoint(train=False)
    
    writer.close()

    ####################
    # what is this ???
    #val_folder = "validation"
    #trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)