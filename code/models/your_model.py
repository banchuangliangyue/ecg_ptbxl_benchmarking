from models.base_model import ClassificationModel
from models.net1d import Net1D
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
import torch.nn as nn
import os, pickle
from tqdm import tqdm
import numpy as np
from time import strftime, gmtime

def get_time_str():
    return strftime("%Y%m%d_%H%M%S", gmtime())

def print_and_log(log_name, my_str):
    out = '{}|{}'.format(get_time_str(), my_str)
    print(out)
    with open(log_name, 'a') as f_log:
        print(out, file=f_log)

def save_checkpoint(state, path):
    filename = 'checkpoint_{.4f}.pth'.format(state['val_auc'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)

def reshape_input(arr):
    new_arr = []
    for i in range(arr.shape[0]):
        new_arr.append(arr[i].T)
    return np.array(new_arr)

def get_model_savedir(dir):
    lst = os.listdir(dir)
    new_lst = []
    for filename in lst:
        if filename.endswith('.pth'):
            new_lst.append(filename)
    new_lst.sort()
    return new_lst[-1]

class YourModel(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape 

    def fit(self, X_train, y_train, X_val, y_val):
        gpu_id = 1

        ### no need to change
        batch_size = 128
        lr = 1e-3
        weight_decay = 1e-4
        early_stop_lr = 1e-5
        Epochs = 10
        eval_steps = 20

        ### data
        trainset_x, trainset_y = X_train, y_train
        valset_x, valset_y = X_val, y_val

        device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

        ### model net1d
        model = Net1D(in_channels=12,base_filters=64,ratio=1,filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],kernel_size=16,stride=2,groups_width=16,verbose=False,
            use_bn=False,use_do=False,n_classes=self.n_classes)

        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)

        ### train model
        best_val_auroc = 0.
        step = 0

        for epoch in range(Epochs):

            ### train
            for idx in tqdm(range(0, trainset_x.shape[0], batch_size), desc='Training'):


                input_x, input_y = trainset_x[idx:idx+batch_size], trainset_y[idx:idx+batch_size]
                input_x = reshape_input(input_x)
                input_x = torch.from_numpy(input_x).to(device).float()
                input_y = torch.from_numpy(input_y).to(device).float()

                outputs = model(input_x)
                loss = criterion(outputs, input_y)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                step += 1

                if step % eval_steps == 0:
                    # val
                    model.eval()
                    all_gt = []
                    all_pred_prob = []
                    with torch.no_grad():
                        for idx in tqdm(range(valset_x.shape[0]), desc='Validating'):
                            input_x, input_y = valset_x[idx], valset_y[idx]
                            input_x = reshape_input(input_x)
                            input_x = torch.from_numpy(input_x).to(device).float()
                            input_y = torch.from_numpy(input_y).to(device).float()
                            logits = model(input_x)
                            pred = torch.sigmoid(logits)
                            all_pred_prob.append(pred.cpu().data.numpy())
                            all_gt.append(input_y.cpu().data.numpy())
                    all_pred_prob = np.concatenate(all_pred_prob)
                    all_gt = np.concatenate(all_gt)
                    all_gt = np.array(all_gt)

                    val_auroc = roc_auc_score(all_gt, all_pred_prob)

                    # save train_log
                    log_str = 'Epoch {} step {}, val: {:.4f}'.format(epoch, step, val_auroc)
                    log_name = os.path.join(self.outputfolder, 'log.txt')
                    print_and_log(log_name, log_str)

                    # save model and res
                    saved_dir = self.outputfolder
                    is_best = bool(val_auroc > best_val_auroc)
                    if is_best:
                        best_val_auroc = val_auroc
                        print('==> Saving a new val best!')
                        save_checkpoint({
                            'epoch': epoch,
                            'step': step,
                            'state_dict': model.state_dict()
                        }, saved_dir)

                    scheduler.step(val_auroc)
                    ### early stop
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr < early_stop_lr:
                        print("Early stop")
                        exit()

                    model.train()  # set back to train

    def predict(self, X):
        gpu_id = 1
        batch_size = 128
        device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

        ### model net1d
        model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                      m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
                      use_bn=False, use_do=False, n_classes=self.n_classes)

        model_savedir = get_model_savedir(self.outputfolder)
        checkpoint = torch.load(self.outputfolder + model_savedir)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        model.eval()
        all_pred_prob = []
        with torch.no_grad():
            for idx in tqdm(range(0, X.shape[0], batch_size), desc='Predicting'):
                input_x = X[idx:idx+batch_size]
                input_x = reshape_input(input_x)
                input_x = torch.from_numpy(input_x).to(device).float()
                logits = model(input_x)
                pred = torch.sigmoid(logits)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)

        return all_pred_prob


