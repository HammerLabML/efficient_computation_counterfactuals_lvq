#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import time
random.seed(42)
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_lvq import GlvqModel, GmlvqModel, LgmlvqModel

from counterfactuals_lvq import LvqCounterfactual, MatrixLvqCounterfactual, LocalizedMatrixLvqCounterfactual


def encode_labels(y_test, y_pred):
    enc = OneHotEncoder()
    enc.fit(y_test)

    return enc.transform(y_test).toarray(), enc.transform(y_pred).toarray()


if __name__ == "__main__":
    # Load data
    X, y = load_iris(return_X_y=True)
    X, y = shuffle(X, y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4242)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit model
    model = GlvqModel
    #model = GmlvqModel # Uncomment this line if you want to use the Generalized Matrix Learning Veqtor Quantization model
    #model = LgmlvqModel  # Uncomment this line if you want to use the LGMLVQ model

    model = model(prototypes_per_class=2, random_state=4242)
    
    model.fit(X_train, y_train)

    # Evaluation
    y_test_pred = model.predict(X_test)
    y_test_, y_test_pred_ = encode_labels(y_test.reshape(-1, 1), y_test_pred.reshape(-1, 1))
    print("ROC-AUC: {0}\n".format(roc_auc_score(y_test_, y_test_pred_, average="weighted")))

    # Compute counterfactuals
    features_whitelist = None

    md = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - md), axis=0)

    i = 0
    x_orig = X_test[i,:]
    y_orig_pred = model.predict([x_orig])
    print("Prediction of the original data point: {0}\n Ground truth: {1}".format(y_orig_pred, y_test[i]))

    y_target = 0
    cf = None
    if isinstance(model, GlvqModel):
        cf = LvqCounterfactual(model)
    elif isinstance(model, GmlvqModel):
        cf = MatrixLvqCounterfactual(model)
    elif isinstance(model, LgmlvqModel):
        cf = LocalizedMatrixLvqCounterfactual(model)

    xcf, ycf, delta = cf.generate_counterfactual(x_orig, y_target, features_whitelist, mad)
    print("Counterfactual: {0}\n    Original: {1}".format(xcf, x_orig))
    print("Prediction of counterfactual: {0}\n  Prediction of original: {1}".format(ycf, y_orig_pred))
    print("Norm of change: {0}".format(np.linalg.norm(x_orig - xcf, 1)))

