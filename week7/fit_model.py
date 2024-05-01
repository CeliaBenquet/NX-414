import numpy as np
import argparse
import torch
import torch.nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import h5py
import os
import torch

from collections import OrderedDict

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import train_test_split

import cebra
import sys
import pathlib
import time

sys.path.append("./../")

import logging

log = logging.getLogger(__name__)

import models
import week6.utils

#print(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=200)
parser.add_argument("--model_architecture", type=str, default="cnn-2")
parser.add_argument("--num_iterations", type=int, default=10000)
parser.add_argument("--cpus-per-task", type=float, default=4)
parser.add_argument("--gpus-per-task", type=float, default=1)
parser.add_argument("--datapath", type=str, default="")
parser.add_argument("--logdir", type=str, default="checkpoints_")
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=100) # for CEBRA only 
parser.add_argument("--batch_size", type=int, default=64) # for CEBRA only
parser.add_argument("--device", type=str, default="cuda")

ARGS = parser.parse_args()

# Create log directory
LOGDIR = pathlib.Path(ARGS.logdir) / time.strftime("%Y-%m-%d-%H%M%S")
if pathlib.Path.exists(LOGDIR):
    assert len(pathlib.Path.iterdir(LOGDIR)) == 0
else:
    LOGDIR.mkdir(parents=True, exist_ok=True)

def choose_model(model_name, num_neurons):
    if model_name == "cnn-2":
        model = models.TwoLayersCNN(num_neurons=num_neurons).to(ARGS.device)
    elif model_name == "cnn-3":
        model = models.ThreeLayersCNN(num_neurons=num_neurons).to(ARGS.device)
    elif model_name == "cebra_time":
        model = cebra.CEBRA(max_iterations=ARGS.num_iterations, device=ARGS.device)
    else:
        NotImplementedError(f"{model_name}")
    return model


def analyze_model(outputs, target):
    scores = {}

    scores["mse"] = mean_squared_error(outputs, target)
    scores["r2"] = r2_score(outputs, target)
    scores["explained_variance"] = explained_variance_score(outputs, target)

    return scores


def train_model(
    dataloader,
    valid_dataloader,
    model,
    optimizer,
    criterion,
    num_epochs,
    log,
    max_patience=50,
    device="cpu",
):
    train_losses = []
    val_losses = []
    
    best_loss = 1e3

    for epoch in tqdm(range(num_epochs)):
        train_epoch_loss = 0
        val_epoch_loss = 0
        
        # Training
        model.train()
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item() * data.size(0)

            # Compute all metrics
            train_metrics = analyze_model(
                outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
            )
            
        train_epoch_loss = train_epoch_loss/len(dataloader.dataset)
        train_losses.append(train_epoch_loss)

        # Evaluation
        model.eval()
        with torch.no_grad():
            for data, labels in valid_dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_epoch_loss += loss.item() * data.size(0)

                # Compute all metrics
                val_metrics = analyze_model(
                    outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
                )

            val_epoch_loss = val_epoch_loss/len(dataloader.dataset)
            val_losses.append(val_epoch_loss)

        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch}, Train loss: {train_epoch_loss}; "
                f"Val loss: {val_epoch_loss}"
            )
            for key in train_metrics.keys():
                log.info(f"Train {key}: {np.array(train_metrics[key]).mean()}")
                log.info(f"Val {key}: {np.array(val_metrics[key]).mean()}")
                
        if val_epoch_loss < best_loss:
            best_model_state_dict = {k:v.detach().to('cpu') for k, v in model.state_dict().items()}
            best_model_state_dict = OrderedDict(best_model_state_dict)
            best_loss = val_epoch_loss
            patience = 0
        else:
            patience+=1

        if patience == max_patience:
            break

    log.info(
        f"Final epoch, Train loss: {train_epoch_loss/len(dataloader)}; "
        f"Val loss: {val_epoch_loss/len(valid_dataloader)}"
    )

    for key in train_metrics.keys():
        log.info(f"Train {key}: {np.array(train_metrics[key]).mean()}")
        log.info(f"Val {key}: {np.array(val_metrics[key]).mean()}")

    plt.plot(train_losses[1:])
    #plt.plot(val_losses[1:])
    plt.savefig(LOGDIR / "loss.png")

    return model, val_metrics, train_metrics


def fit_rectlin(train_input, eval_input, train_output, alpha=0.0):
    # Fit linear regression
    lr = Ridge(alpha=alpha)
    lr.fit(train_input, train_output)
    train_pred = lr.predict(train_input)
    eval_pred = lr.predict(eval_input)

    # Rectify to prevent negative or 0 rate predictions
    train_pred[train_pred < 1e-10] = 1e-10
    eval_pred[eval_pred < 1e-10] = 1e-10

    return train_pred, eval_pred


class NeuralDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def fit_model():

    (
        stimulus_train,
        stimulus_val,
        stimulus_test,
        objects_train,
        objects_val,
        objects_test,
        spikes_train,
        spikes_val,
    ) = week6.utils.load_it_data(ARGS.datapath)

    # Define data
    train_dataset = NeuralDataset(stimulus_train, spikes_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=4
    )

    valid_dataset = NeuralDataset(stimulus_val, spikes_val)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=4
    )

    # Define model
    model = choose_model(ARGS.model_architecture, num_neurons=spikes_train.shape[1])

    # TODO: get dino features, spit training data
    if ARGS.model_architecture == "cebra_time":
        model.fit(spikes_train)

        embedding_train = model.transform(spikes_train)
        embedding_val = model.transform(spikes_val)

        train_pred, val_pred = fit_rectlin(embedding_train, embedding_val, spikes_train)
        train_metrics = analyze_model(train_pred, spikes_train)
        val_metrics = analyze_model(val_pred, spikes_val)

    else:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.learning_rate)

        model, val_metrics, train_metrics = train_model(
            dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=ARGS.num_epochs,
            log=log,
            device=ARGS.device,
        )

    for key in val_metrics.keys():
        log.info(f"Train {key}: {np.array(train_metrics[key]).mean()}")
        log.info(f"Val {key}: {np.array(val_metrics[key]).mean()}")
        
    torch.save(model, LOGDIR / "model.pt")


if __name__ == "__main__":

    logging.basicConfig(filename=LOGDIR / f"logging.log", level=logging.INFO)
    log.info(ARGS)
    log.info("Started")

    metrics = fit_model()
