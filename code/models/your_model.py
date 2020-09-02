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
import pandas as pd


def get_time_str():
    return strftime("%Y%m%d_%H%M%S", gmtime())


def print_and_log(log_name, my_str):
    out = '{}|{}'.format(get_time_str(), my_str)
    print(out)
    with open(log_name, 'a') as f_log:
        print(out, file=f_log)


def save_checkpoint(state, path):
    filename = 'checkpoint_{:.4f}.pth'.format(state['val_auroc'])
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


def aggreate_predict(all_pid, gt, prob):
    # group by pid
    final_pred = []
    final_gt = []
    pid_set = set(all_pid)
    for pid in pid_set:
        select_idx = (all_pid == pid)
        tmp_pred = prob[select_idx]
        tmp_gt = gt[select_idx]

        final_pred.append(np.max(tmp_pred, axis=0))
        final_gt.append(tmp_gt[0])

    return np.array(final_gt), np.array(final_pred)


class YourModel(ClassificationModel):
    def __init__(self, name, n_classes, sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.gpu_id = 6
        self.model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                      m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
                      use_bn=True, use_do=True, n_classes=self.n_classes)
        self.batch_size = 1024
        self.device = torch.device('cuda:{}'.format(self.gpu_id) if torch.cuda.is_available() else 'cpu')


    def fit(self, X_train, y_train, X_val, y_val, pid_val):

        ### no need to change
        lr = 1e-3
        weight_decay = 1e-4
        early_stop_lr = 1e-5
        Epochs = 50
        eval_steps = 5

        ### data
        trainset_x, trainset_y = X_train, y_train
        valset_x, valset_y = X_val, y_val
        valset_pid = pid_val
        print(trainset_x.shape, trainset_y.shape)
        print(valset_x.shape, valset_y.shape)

        ### model net1d
        model = self.model

        model.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)

        ### train model
        best_val_auroc = 0.
        step = 0

        all_res = []
        for epoch in range(Epochs):

            ### train
            for idx in tqdm(range(0, trainset_x.shape[0], self.batch_size), desc='Training'):
                input_x, input_y = trainset_x[idx:idx + self.batch_size], trainset_y[idx:idx + self.batch_size]
                # input_x = np.expand_dims(input_x.T, axis=0)
                # input_y = np.expand_dims(input_y, axis=0)
                input_x = reshape_input(input_x)
                input_x = torch.from_numpy(input_x).to(self.device).float()
                input_y = torch.from_numpy(input_y).to(self.device).float()
                # print(input_x.shape)
                outputs = model(input_x)
                loss = criterion(outputs, input_y)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                step += 1

                if step % eval_steps == 0:
                    train_loss = loss.cpu().data.numpy()
                    # val
                    model.eval()
                    all_gt = []
                    all_pred_prob = []
                    val_loss = []
                    with torch.no_grad():
                        for idx in tqdm(range(0, valset_x.shape[0], self.batch_size), desc='Validating'):
                            input_x, input_y = valset_x[idx:idx + self.batch_size], valset_y[idx:idx + self.batch_size]
                            input_x = reshape_input(input_x)
                            # input_x = np.expand_dims(input_x.T, axis=0)
                            # input_y = np.expand_dims(input_y, axis=0)
                            input_x = torch.from_numpy(input_x).to(self.device).float()
                            input_y = torch.from_numpy(input_y).to(self.device).float()
                            # print(input_x.shape)
                            logits = model(input_x)
                            v_loss = criterion(logits, input_y)
                            val_loss.append(v_loss.cpu().data.numpy())

                            pred = torch.sigmoid(logits)
                            all_pred_prob.append(pred.cpu().data.numpy())
                            all_gt.append(input_y.cpu().data.numpy())
                    all_pred_prob = np.concatenate(all_pred_prob)
                    all_gt = np.concatenate(all_gt)
                    all_gt = np.array(all_gt)
                    val_loss = np.array(val_loss).mean()
                    # print(all_pred_prob.shape)
                    all_gt, all_pred_prob = aggreate_predict(valset_pid, all_gt, all_pred_prob)
                    # print(all_pred_prob.shape)
                    val_auroc = roc_auc_score(all_gt, all_pred_prob)

                    # save train_log
                    log_str = 'Epoch {} step {}, val_auroc: {:.4f}'.format(epoch, step, val_auroc)
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
                            'state_dict': model.state_dict(),
                            'val_auroc': val_auroc
                        }, saved_dir)

                    scheduler.step(val_auroc)
                    ### early stop
                    current_lr = optimizer.param_groups[0]['lr']
                    all_res.append([epoch, step, train_loss, val_loss, current_lr])
                    df = pd.DataFrame(all_res, columns=['epoch', 'step', 'train_loss', 'val_loss', 'lr'])
                    df.to_csv(os.path.join(self.outputfolder, 'res.csv'), index=False, float_format='%.5f')

                    # if current_lr < early_stop_lr:
                    # print("Early stop")
                    # exit()

                    model.train()  # set back to train

    def predict(self, X, pid):


        model_savedir = get_model_savedir(self.outputfolder)
        checkpoint = torch.load(self.outputfolder + model_savedir)
        model = self.model
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)

        model.eval()
        all_pred_prob = []
        with torch.no_grad():
            for idx in tqdm(range(0, X.shape[0], self.batch_size), desc='Predicting'):
                input_x = X[idx:idx + self.batch_size]
                input_x = reshape_input(input_x)
                input_x = torch.from_numpy(input_x).to(self.device).float()
                logits = model(input_x)
                pred = torch.sigmoid(logits)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        # print('pred on test de shape: ', all_pred_prob.shape)
        pid_set = set(pid)
        final_pred = []
        for p in pid_set:
            select_idx = (pid == p)
            tmp_pred = all_pred_prob[select_idx]
            final_pred.append(np.max(tmp_pred, axis=0))
        # print(np.array(final_pred).shape)
        return np.array(final_pred)




