import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        results = []
        truths = []
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
            batch_size = text.size(0)
            preds = model(text, audio, vision)
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            results.append(preds)  # [batch,1]
            truths.append(eval_attr)  # [batch, 1]
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                batch_size = text.size(0)
                preds = model(text, audio, vision)
                total_loss += criterion(preds, eval_attr).item() * batch_size
                results.append(preds)       # [batch,1]
                truths.append(eval_attr)    # [batch, 1]
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_results, train_truths = train(model, optimizer, criterion)
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        test_loss, test_results, test_truths = evaluate(model, criterion, test=True)
        train_ac = eval_mosi_short(train_results, train_truths, True)
        val_ac = eval_mosi_short(val_results, val_truths, True)
        test_acc = eval_mosi_short(test_results, test_truths, True)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Acc {:5.4f} '.format(epoch, duration,train_ac))
        print('Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(val_loss, test_loss))
        print('Valid Acc  {:5.4f} | Test Acc  {:5.4f}'.format(val_ac, test_acc))
        print("-"*50)
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss
    model = load_model(hyp_params, model, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=True)
    eval_mosi(results, truths, True)