import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
import time
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

if __name__ == '__main__':
    # start_time = time.time()
    # # ## Constants (parameters) initialization
    # device_id = [0,1,2,3]
    # num_workers = 4
    # batch_size = 128

    # # add our package dir to path 
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'codes/VGG_BatchNorm/figures')
    # models_path = os.path.join(home_path, 'codes/VGG_BatchNorm/saved_models')

    # # Make sure you are using the right device.
    # device_id = device_id
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
    # print(device)
    # if torch.cuda.is_available():
    #     print(torch.cuda.get_device_name(3))
    # else:
    #     print("CUDA is not available. Using CPU.")


    # # Initialize your data loader and
    # # make sure that dataloader works
    # # as expected by observing one
    # # sample from it.
    # train_loader = get_cifar_loader(train=True)
    # val_loader = get_cifar_loader(train=False)
    # for X, y in train_loader:
    #     ## --------------------
    #     # Add code as needed
    #     plt.imshow(X[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    #     print(f"Label: {y[0]}")
    #     plt.savefig(os.path.join(figures_path, 'sample.png'))
    #     print(f"Images shape: {X[0].shape}, Labels shape: {y[0].shape}")
    #     ## --------------------
    #     break



    # # This function is used to calculate the accuracy of model classification
    # def get_accuracy(model, data_loader):
    #     ## --------------------
    #     # Add code as needed
    #     model.eval()
    #     running_acc = 0.0
    #     total = 0
    #     with torch.no_grad():
    #         for images, labels in data_loader:
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             outputs = model(images)
    #             _, preds = torch.max(outputs, 1)

    #             running_acc += torch.sum(preds == labels).item()
    #             total += labels.size(0)
    #     return running_acc / total
    #     ## --------------------


    # # Set a random seed to ensure reproducible results
    # def set_random_seeds(seed_value=0, device='cpu'):
    #     np.random.seed(seed_value)
    #     torch.manual_seed(seed_value)
    #     random.seed(seed_value)
    #     if device != 'cpu': 
    #         torch.cuda.manual_seed(seed_value)
    #         torch.cuda.manual_seed_all(seed_value)
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False


    # # We use this function to complete the entire
    # # training process. In order to plot the loss landscape,
    # # you need to record the loss value of each step.
    # # Of course, as before, you can test your model
    # # after drawing a training round and save the curve
    # # to observe the training
    # def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    #     model.to(device)
    #     learning_curve = [np.nan] * epochs_n
    #     train_accuracy_curve = [np.nan] * epochs_n
    #     val_accuracy_curve = [np.nan] * epochs_n
    #     max_val_accuracy = 0
    #     max_val_accuracy_epoch = 0

    #     batches_n = len(train_loader)
    #     losses_list = []
    #     grads = []
    #     for epoch in tqdm(range(epochs_n), unit='epoch'):
    #         if scheduler is not None:
    #             scheduler.step()
    #         model.train()

    #         loss_list = []  # use this to record the loss value of each step
    #         # grad = []  # use this to record the loss gradient of each step
    #         learning_curve[epoch] = 0  # maintain this to plot the training curve

    #         for data in train_loader:
    #             x, y = data
    #             x = x.to(device)
    #             y = y.to(device)
    #             optimizer.zero_grad()
    #             prediction = model(x)
    #             loss = criterion(prediction, y)
    #             # You may need to record some variable values here
    #             # if you want to get loss gradient, use
    #             # grad = model.classifier[4].weight.grad.clone()
    #             ## --------------------
    #             # Add your code
    #             loss_list.append(loss.item())
    #             learning_curve[epoch] += loss.item()
    #             ## --------------------


    #             loss.backward()
    #             # grad = model.classifier[4].weight.grad.clone()
    #             optimizer.step()

    #         losses_list.append(loss_list)
    #         # grads.append(grad)
    #         display.clear_output(wait=True)
    #         f, axes = plt.subplots(1, 2, figsize=(15, 5))

    #         learning_curve[epoch] /= batches_n
    #         axes[0].plot(learning_curve, label='Training Loss')
    #         axes[0].legend()
    #         axes[0].set_title('Loss')
    #         axes[0].set_xlabel('Epochs')
    #         axes[0].set_ylabel('Loss')

    #         # Test your model and save figure here (not required)
    #         # remember to use model.eval()
    #         ## --------------------
    #         # Add code as needed
    #         model.eval()
    #         train_accuracy = get_accuracy(model, train_loader)
    #         val_accuracy = get_accuracy(model, val_loader)
    #         train_accuracy_curve[epoch] = train_accuracy
    #         val_accuracy_curve[epoch] = val_accuracy
    #         axes[1].plot(train_accuracy_curve, label='Train Accuracy')
    #         axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
    #         axes[1].legend()
    #         axes[1].set_title('Accuracy')
    #         axes[1].set_xlabel('Epochs')
    #         axes[1].set_ylabel('Accuracy')
    #         plt.savefig(os.path.join(figures_path, 'loss_acc_landscape.png'))
    #         plt.close()

    #         if val_accuracy > max_val_accuracy:
    #             max_val_accuracy = val_accuracy
    #             max_val_accuracy_epoch = epoch
    #             if best_model_path is not None:
    #                 torch.save(model.state_dict(), best_model_path)
    #         ## --------------------
    #     print(f"Best Validation Accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch}")
    #     return losses_list, grads


    # # Train your model
    # # feel free to modify
    # epo = 20
    # loss_save_path = './VGG_BatchNorm/losses'
    # grad_save_path = './VGG_BatchNorm/grads'

    # set_random_seeds(seed_value=2020, device=device)

    # model = VGG_A()
    # lr = 5e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # criterion = nn.CrossEntropyLoss()
    # loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    # np.savetxt(os.path.join(loss_save_path, 'loss5e-4.txt'), loss, fmt='%s', delimiter=' ')
    # # np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

    # # model = VGG_A_BatchNorm()
    # # lr = 5e-4
    # # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # # criterion = nn.CrossEntropyLoss()
    # # loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    # # np.savetxt(os.path.join(loss_save_path, 'loss5e-4BN.txt'), loss, fmt='%s', delimiter=' ')
    # # np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Total execution time: {elapsed_time / 60:.2f} minutes")

    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    min_curve = []
    max_curve = []
    ## --------------------
    # Add your code
    lrs = ['1e-3', '2e-3', '1e-4', '5e-4']
    loss_files = [f'VGG_BatchNorm/losses/loss{lr}.txt' for lr in lrs]
    all_losses = []
    for file in loss_files:
        loss_data = np.loadtxt(file)
        all_losses.append(loss_data.flatten())
    num_steps = len(all_losses[0])
    for step in range(0, num_steps, 20):
        step_losses = [losses[step] for losses in all_losses]
        min_curve.append(min(step_losses))
        max_curve.append(max(step_losses))
    ## --------------------

    # Use this function to plot the final loss landscape,
    # fill the area between the two curves can use plt.fill_between()
    def plot_loss_landscape(min_curve, max_curve, label):
        ## --------------------
        # Add your code
        # plt.plot(min_curve)
        # plt.plot(max_curve)
        pass
        ## --------------------


    plt.figure(figsize=(12, 8))
    plt.fill_between(range(len(min_curve)), min_curve, max_curve, color='lightgreen', edgecolor='green', alpha=0.5, label="Standard VGG") 

    min_curve = []
    max_curve = []
    bn_loss_files = [f'VGG_BatchNorm/losses/loss{lr}BN.txt' for lr in lrs]
    all_losses = []
    for file in bn_loss_files:
        loss_data = np.loadtxt(file)
        all_losses.append(loss_data.flatten())
    num_steps = len(all_losses[0])
    for step in range(0, num_steps, 20):
        step_losses = [losses[step] for losses in all_losses]
        min_curve.append(min(step_losses))
        max_curve.append(max(step_losses))

    plt.fill_between(range(len(min_curve)), min_curve, max_curve, color='lightcoral', edgecolor='red', alpha=0.5, label="Standard VGG + BatchNorm") 
    plt.title("Loss Landscape")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.ylim((0, 2.5))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_path, "loss_landscape.png"))