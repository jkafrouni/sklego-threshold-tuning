import pandas as pd
import numpy as np

from sklego.datasets import load_hearts
from sklearn.model_selection import train_test_split

data = load_hearts(as_frame=True)
X, y = data.drop(columns='target'), data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)