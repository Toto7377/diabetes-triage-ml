from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

SEED = 42

def load_data():
    Xy = load_diabetes(as_frame=True)
    df = Xy.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=SEED, shuffle=True)
