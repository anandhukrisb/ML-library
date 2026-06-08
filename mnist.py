import numpy as np
from urllib import request
import gzip
import os

def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        request.urlretrieve(url, filename)

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.float32)

base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
download(base + "train-images-idx3-ubyte.gz", "train-images.gz")
download(base + "train-labels-idx1-ubyte.gz", "train-labels.gz")
download(base + "t10k-images-idx3-ubyte.gz",  "test-images.gz")
download(base + "t10k-labels-idx1-ubyte.gz",  "test-labels.gz")

train_images = load_images("train-images.gz")
train_labels = load_labels("train-labels.gz")
test_images  = load_images("test-images.gz")
test_labels  = load_labels("test-labels.gz")

print(train_images.shape)
print(train_labels.shape)