# import matplotlib.pyplot as plt

# def parse_log_file(log_file):
#     epochs = []
#     losses = []
#     accuracies = []

#     with open(log_file, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith('transformer2-epoch='):
#                 parts = line.split(',')
#                 epoch = int(parts[0].split('=')[1])
#                 loss = float(parts[1].split('=')[1])
#                 acc = float(parts[2].split('=')[1])
#                 epochs.append(epoch)
#                 losses.append(loss)
#                 accuracies.append(acc)

#     return epochs, losses, accuracies

# def plot_training_progress(epochs, losses, accuracies):
#     # Get initial values for each epoch
#     unique_epochs = list(set(epochs))
#     initial_losses = [losses[epochs.index(epoch)] for epoch in unique_epochs]
#     initial_accuracies = [accuracies[epochs.index(epoch)] for epoch in unique_epochs]

#     # Plot loss curve
#     plt.figure(figsize=(10, 5))
#     plt.plot(unique_epochs, initial_losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.grid(True)
#     plt.show()

#     # Plot accuracy curve
#     plt.figure(figsize=(10, 5))
#     plt.plot(unique_epochs, initial_accuracies)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Training Accuracy Curve')
#     plt.grid(True)
#     plt.show()

# # Log file path
# log_file = '/home/lizijian/lincode/ember/DL/log_draw1.txt'

# # Parse the log file
# epochs, losses, accuracies = parse_log_file(log_file)

# # Plot the training progress
# plot_training_progress(epochs, losses, accuracies)


import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file):
    epochs = []
    losses = []
    accuracies = []

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('transformer4-epoch='):
                parts = line.split(',')
                epoch = int(parts[0].split('=')[1])
                loss = float(parts[1].split('=')[1])
                acc = float(parts[2].split('=')[1])
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(acc)

    return epochs, losses, accuracies

def plot_training_progress(epochs, losses, accuracies):
    unique_epochs = np.unique(epochs)
    avg_losses = [np.mean([losses[i] for i, epoch in enumerate(epochs) if epoch == unique_epoch]) for unique_epoch in unique_epochs]
    avg_accuracies = [np.mean([accuracies[i] for i, epoch in enumerate(epochs) if epoch == unique_epoch]) for unique_epoch in unique_epochs]

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(unique_epochs, avg_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    plt.grid(True)
    plt.savefig('t4_loss_curve.jpg', dpi=600) 
    plt.show()

    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(unique_epochs, avg_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Average Training Accuracy per Epoch')
    plt.grid(True)
    plt.savefig('t4_acc_curve.jpg', dpi=600) 
    plt.show()

# Log file path
log_file = '/home/lizijian/lincode/ember/DL/log_draw3.txt'

# Parse the log file
epochs, losses, accuracies = parse_log_file(log_file)

# Plot the training progress
plot_training_progress(epochs, losses, accuracies)



