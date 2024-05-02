import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score

import sys
import pathlib
import time

sys.path.append("./../")

import models
import utils


class NeuralDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Session:
    def __init__(
        self,
        model_architecture: str,
        datapath: str,
        logdir: str,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        batch_size: int = 64,
        device: str = "cuda",
    ):
        self.set_log_dir(logdir)
        self.device = device
        self.model_architecture = model_architecture
        self.datapath = datapath
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        print(f"Model architecture: {self.model_architecture}, "
              f"Batch size: {self.batch_size}, "
              f"Num epochs: {self.num_epochs}, "
              f"Learning rate: {self.learning_rate}"
              )


    def set_log_dir(self, logdir):
        "Create the log folder." ""
        self.logdir = pathlib.Path(logdir) / time.strftime("%Y-%m-%d-%H%M%S")
        if pathlib.Path.exists(self.logdir):
            assert len(pathlib.Path.iterdir(self.logdir)) == 0
        else:
            self.logdir.mkdir(parents=True, exist_ok=True)

    def set_model(self, model_architecture, num_neurons):
        if model_architecture == "cnn-2":
            model = models.TwoLayersCNN(num_neurons=num_neurons).to(self.device)
        elif model_architecture == "cnn-3":
            model = models.ThreeLayersCNN(num_neurons=num_neurons).to(self.device)
        else:
            NotImplementedError(f"{model_architecture}")
        self.model = model


    def analyze_model(self, outputs, target):
        scores = {}

        scores["mse"] = mean_squared_error(outputs, target)
        scores["r2"] = r2_score(outputs, target)
        scores["explained_variance"] = explained_variance_score(outputs, target)

        return scores


    def fit_model(self):
        (
            stimulus_train,
            stimulus_val,
            _,
            _,
            _,
            _,
            spikes_train,
            spikes_val,
        ) = utils.load_it_data(self.datapath)

        ### Define the dataloaders
        train_dataset = NeuralDataset(stimulus_train, spikes_train)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        valid_dataset = NeuralDataset(stimulus_val, spikes_val)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        ### Define the model
        self.set_model(
            self.model_architecture, num_neurons=spikes_train.shape[1]
        )

        ### Define the loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        ### Training
        self.train_model(
            dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=criterion,
        )

        ### Evaluation
        val_metrics = self.evaluate_model(dataloader=valid_dataloader)

        for key in val_metrics.keys():
            print(f"Val {key}: {np.array(val_metrics[key]).mean()}")

        torch.save(self.model, self.logdir / "model.pt")

    def train_model(
        self,
        dataloader,
        valid_dataloader,
        optimizer,
        criterion,
        max_patience=60,
    ):
        train_losses = []
        val_losses = []

        best_loss = 1e3
        best_model = self.model

        for epoch in tqdm(range(self.num_epochs)):
            train_epoch_loss = 0
            val_epoch_loss = 0

            # Training
            self.model.train()
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item() * data.size(0)

            train_epoch_loss = train_epoch_loss / len(dataloader.dataset)
            train_losses.append(train_epoch_loss)

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                for data, labels in valid_dataloader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                    val_epoch_loss += loss.item() * data.size(0)

            val_epoch_loss = val_epoch_loss / len(valid_dataloader.dataset)
            val_losses.append(val_epoch_loss)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Train loss: {train_epoch_loss}; Val loss: {val_epoch_loss}"
                )

            if val_epoch_loss < best_loss:
                best_model = self.model
                best_loss = val_epoch_loss
                patience = 0
            else:
                patience += 1

            if patience == max_patience:
                print(
                    f"Final epoch {epoch}: Train loss: {train_epoch_loss}; Val loss: {val_epoch_loss}"
                )
                break
        
        print(
            f"Final epoch {epoch}: Train loss: {train_epoch_loss}; Val loss: {val_epoch_loss}"
        )
        
        self.model = best_model
        
        plt.plot(train_losses[1:])
        plt.savefig(self.logdir / "loss.png")
        plt.close()

    def evaluate_model(self, dataloader):
        self.model.eval()
        predictions = []
        true_labels = []
        for data, labels in dataloader:
            data, labels = data.to(self.device), labels.to(self.device)
            outputs = self.model(data)
            predictions.append(outputs.detach().cpu())
            true_labels.append(labels.detach().cpu())
            
        y_pred = torch.cat(predictions, dim=0).numpy()
        y_true = torch.cat(true_labels, dim=0).numpy()

        return self.analyze_model(
            y_pred, y_true
        )
