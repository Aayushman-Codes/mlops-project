import os
import pandas as pd


def test_dataset_exists():
    assert os.path.exists("data/iris.csv")


def test_dataset_has_correct_columns():
    data = pd.read_csv("data/iris.csv")

    expected_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    assert list(data.columns) == expected_columns


def test_dataset_is_not_empty():
    data = pd.read_csv("data/iris.csv")

    assert len(data) > 0
