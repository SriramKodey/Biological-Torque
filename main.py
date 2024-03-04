import loader
import model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
import torch

def main(path):
    df = loader.loadData(path, header=False)
    df = pd.DataFrame(df.values)
    df = df.dropna(axis=0)
    data = df.to_numpy()
    x = data[:, 0:-3]
    y = data[:, -3:]
    print(np.shape(x), np.shape(y))
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    print(np.shape(y), np.shape(x))

    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(np.shape(X_train), np.shape(y_train))

    neuralNet = model.NeuralNet(np.shape(X_train)[1], np.shape(y_train)[1]).to(model.device)
    print(neuralNet)

    model.train(neuralNet, X_train, y_train)

    model.test(neuralNet, X_test, y_test)

    x_test_2 = torch.tensor(X_test[0:2, :], dtype=torch.float32).to(model.device)
    pred = neuralNet(x_test_2)
    print(y_test[0:2, :], pred)

if __name__ == "__main__":
    main("C://Users/kodey/Documents/546_Dataset/10_09_18/levelground/emg//levelground_cw_normal_04_01_features.csv")