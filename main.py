from model import NeuralNetwork
import numpy as np
import h5py

def load_data(path):

    data_hf = h5py.File(path, 'r')
    x_train = np.array(data_hf['x_train'])
    y_train = np.array(data_hf['y_train'][:, 0])
    x_test = np.array(data_hf['x_test'])
    y_test = np.array(data_hf['y_test'][:, 0])
    data_hf.close()
    return x_train, y_train, x_test, y_test

def main():

    # Load dataset.
    data = load_data("MNISTdata.hdf5")
    train_data, test_data = data[:2], data[2:]

    nn = NeuralNetwork(train_data[0].shape[1], hidden_units=100)

    # Train.
    print("training Network...")
    nn.train(train_data, learning_rate1=0.1, epochs=20)

    # Test
    print("testing Network...")
    nn.test(test_data)

main()