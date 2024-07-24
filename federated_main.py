import numpy as np
import torch
import torch.nn as nn
import math
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CifarCNN, CNN_FMNIST, FC_SYN, ResNet50
from options import args_parser
import copy
from data_loader_synthetic import get_dataset_synthetic
from data_loader_medical import get_dataset_medical
import wandb
from svd_tools import initialize_grad_len
import torchvision.models as torch_models
torch.set_num_threads(4)

if __name__ == '__main__':
    args = args_parser()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device
    # print(device)
    # load dataset and user groups
    if 'synthetic' in args.dataset:
        train_loader, test_loader, global_train_loader, global_test_loader = get_dataset_synthetic(args)
    elif args.dataset == 'retina' or args.dataset == 'covid_fl':
        train_loader, test_loader, global_train_loader, global_test_loader = get_dataset_medical(args)
    else:
        train_loader, test_loader, global_train_loader, global_test_loader = get_dataset(args)
    
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
        global_model = CifarCNN(num_classes=args.num_classes).to(args.device)
        # args.lr = 0.02
    elif args.dataset == 'fmnist':
        global_model = CNN_FMNIST().to(args.device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=args.num_classes).to(args.device)
    elif args.dataset in ['synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']:
        args.num_classes = 10
        args.num_users = 30
        global_model = FC_SYN().to(args.device)
    elif args.dataset == 'retina':
        args.num_classes = 2
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(args.device)
    elif args.dataset == 'covid_fl':
        args.num_classes = 3
        args.num_users = 12
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(args.device)
    else:
        raise NotImplementedError()
    
    print(args)
    wandb.init(project="FedAvg", name='FedAvg_synthetic_0.5_0.5', config=args)
    
    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss, train_acc = [], []
    test_acc = []
    local_accs1, local_accs2 = [], []
#====================================== Set the local clients ================================================#
    local_clients = []
    for idx in range(args.num_users):
        if args.dataset == 'covid_fl' or args.dataset == 'retina':
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader,
                                             model=copy.deepcopy(global_model)))
        else:
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx], 
                                            model=copy.deepcopy(global_model)))
            
    # Set global test dataset
    global_test_dataset = []
    if args.dataset == 'covid_fl' or args.dataset == 'retina':
        for idx in range(args.num_users):
            global_test_dataset.append(test_loader)
    else:
        for idx in range(args.num_users):
            global_test_dataset.append(test_loader[idx])

    if args.train_rule == 'FedLD' or args.train_rule == 'FedGH':
        # initialize grad_history and grad_len
        grad_history = {}
        for k in global_model.state_dict().keys():
            grad_history[k] = None
        grad_len = initialize_grad_len(global_model, grad_history)

        grad_history['grad_len'] = grad_len

        for round in range(args.epochs):
            loss1, loss2, loss3, global_acc = train_round_parallel(args, global_model, local_clients, round, global_train_loader, global_test_dataset, grad_history)
            # train_loss.append(loss1)
            # print("Train Loss: {}, {}".format(loss1, loss2))
            # print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
            print("Average Training Loss: {}".format(loss3))
            # print("Average Accuracy Before Local Training: {}%, and After Local Training: {}%".format(local_acc1, local_acc2))
            # local_accs1.append(local_acc1)
            # local_accs2.append(local_acc2)
            # wandb.log({"Local Accuracy Before Local Training": local_acc1, "Local Accuracy After Local Training": local_acc2, "Average Training Loss": loss3}, step=round)
            print('Global Accuracy:{}%'.format(global_acc))
            wandb.log({"Global Accuracy": global_acc}, step=round)
    else:
        for round in range(args.epochs):
            loss1, loss2, loss3, global_acc = train_round_parallel(args, global_model, local_clients, round, global_train_loader, global_test_dataset)
            # train_loss.append(loss1)
            # print("Train Loss: {}, {}".format(loss1, loss2))
            # print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
            print("Average Training Loss: {}".format(loss3))
            # print("Average Accuracy Before Local Training: {}%, and After Local Training: {}%".format(local_acc1, local_acc2))
            # local_accs1.append(local_acc1)
            # local_accs2.append(local_acc2)
            # wandb.log({"Local Accuracy Before Local Training": local_acc1, "Local Accuracy After Local Training": local_acc2, "Average Training Loss": loss3}, step=round)
            print('Global Accuracy:{}%'.format(global_acc))
            wandb.log({'Global Accuracy': global_acc}, step=round)
    
    wandb.finish()