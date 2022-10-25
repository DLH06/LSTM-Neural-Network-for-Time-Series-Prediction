import json
import os
import sys

import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from core.model_torch import LSTMNet
from core.torch_dataset import SP500

logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("logs/process_log.log")


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="True Data")
    plt.plot(predicted_data, label="Prediction")
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="True Data")
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for _ in range(i * prediction_len)]
        plt.plot(padding + data, label="Prediction")
        plt.legend()
    plt.show()


def main():
    configs = json.load(open("config.json", "r"))
    if not os.path.exists(configs["model"]["save_dir"]):
        os.makedirs(configs["model"]["save_dir"])
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/checkpoint", exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataloader
    dataset = SP500()
    train_size = int(configs["data"]["train_test_split"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["training"]["batch_size"],
        num_workers=configs["training"]["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs["training"]["batch_size"],
        num_workers=configs["training"]["num_workers"],
    )

    model = LSTMNet().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(configs["training"]["epochs"]):
        lossTrain = 0
        counterTrain = 0
        model.train()
        for _, (inputs, target) in enumerate(
            tqdm(train_loader, desc="Training epoch {}".format(epoch))
        ):
            target = target.to(DEVICE)
            inputs = inputs.to(DEVICE)

            varOutput = model(inputs)
            loss = criterion(varOutput, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossTrain += loss.item()
            counterTrain += 1

        logger.info("Epoch {} Training loss: {}".format(epoch, lossTrain / counterTrain))

        lossVal = 0
        counterVal = 0
        model.eval()
        with torch.no_grad():
            for _, (inputs, target) in enumerate(
                tqdm(test_loader, desc="Eval epoch {}".format(epoch))
            ):
                target = target.to(DEVICE)
                inputs = inputs.to(DEVICE)

                varOutput = model(inputs)
                loss = criterion(varOutput, target)

                lossVal += loss.item()
                counterVal += 1

            logger.info("Epoch {} Validation loss: {}".format(epoch, lossVal / counterVal))

        torch.save(
            model, "logs/checkpoint/epoch_{}_valloss_{}.pt".format(epoch, lossVal)
        )

        scheduler.step()


if __name__ == "__main__":
    main()
