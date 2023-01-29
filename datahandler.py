import numpy as np
from joblib import Memory
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_mnist(batch_size=64, random_seed=42, reshaped=False):
    def split_into_batches(x, batch_size):
        n_batches = len(x) / batch_size
        x = np.array_split(x, n_batches)
        return np.array(x, dtype=object)

    # To cache the downloaded data
    memory = Memory("./mnist")
    fetch_openml_cached = memory.cache(fetch_openml)
    mnist = fetch_openml_cached("mnist_784")

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        mnist.data, mnist.target, test_size=0.33, random_state=random_seed
    )

    # Normalizes the data
    min_max_scaler = MinMaxScaler()

    # One-Hot encodes the targets
    one_hot_encoder = OneHotEncoder()

    # Split the training data into batches
    X_train = split_into_batches(
        min_max_scaler.fit_transform(np.array(X_train)), batch_size
    )

    #
    X_test = split_into_batches(
        min_max_scaler.fit_transform(np.array(X_test)), batch_size
    )

    X_train_new = []
    X_test_new = []
    if reshaped:
        for i in range(len(X_train)):
            if batch_size == 1:
                X_train_new.append(X_train[i].reshape(1, 28, 28))
            else:
                X_train[i] = X_train[i].reshape(-1, 1, 28, 28)
        if batch_size == 1:
            X_train = np.array(X_train_new)
        for i in range(len(X_test)):
            if batch_size == 1:
                X_test_new.append(X_test[i].reshape(1, 28, 28))
            else:
                X_test[i] = X_test[i].reshape(-1, 1, 28, 28)
        if batch_size == 1:
            X_test = np.array(X_test_new)

    # Turn the targets into Numpy arrays and flatten the array
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # One-Hot encode the training data and split it into batches (same as with the training data)
    one_hot_encoder.fit(y_train)
    y_train = one_hot_encoder.transform(y_train).toarray()
    y_train = split_into_batches(np.array(y_train), batch_size)

    one_hot_encoder.fit(y_test)
    y_test = one_hot_encoder.transform(y_test).toarray()
    y_test = split_into_batches(np.array(y_test), batch_size)

    return X_train, y_train, X_test, y_test
