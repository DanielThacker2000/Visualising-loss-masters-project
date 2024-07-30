# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:37:13 2024

@author: dan
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import (regression_coverage_score, regression_mean_width_score, regression_ssc,regression_ssc_score,coverage_width_based,regression_mwi_score)
from mapie.subsample import Subsample
from mapie.conformity_scores import GammaConformityScore,ResidualNormalisedScore

import generate_samples as gen


#tf.config.run_functions_eagerly(True)

@tf.keras.utils.register_keras_serializable()
def quantile_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

@tf.keras.utils.register_keras_serializable()
def quantile_loss_10(y_true, y_pred):
    return quantile_loss(y_true, y_pred, 0.1)

@tf.keras.utils.register_keras_serializable()
def quantile_loss_50(y_true, y_pred):
    return quantile_loss(y_true, y_pred, 0.5)

@tf.keras.utils.register_keras_serializable()
def quantile_loss_90(y_true, y_pred):
    return quantile_loss(y_true, y_pred, 0.9)

class TensorflowToMapie():
    """
    Class that aimes to make compatible a tensorflow model
    with MAPIE. To do so, this class create fit, predict,
    predict_proba and _sklearn_is_fitted_ attributes to the model.
    """
    def __init__(self, model: Sequential) -> None:
        self.model = model
        self.pred_proba = None
        self.trained_ = False
        self.history = None
        #self.get_params 

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # early_stopping_monitor = EarlyStopping(
        #     monitor='val_loss',
        #     min_delta=0,
        #     patience=10,
        #     verbose=0,
        #     mode='auto',
        #     baseline=None,
        #     restore_best_weights=True
        # )
        
        # self.model = self.model.fit(
        #     X_train, y_train, 
        #     batch_size=64,
        #     epochs=2#, callbacks=[early_stopping_monitor]
        # )
        self.history = self.model.fit(
            X_train, y_train, 
            batch_size=64,
            epochs=25, #callbacks=[early_stopping_monitor],
        )

        self.trained_ = True

    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        if len(preds) == 3:
            preds = np.mean(preds, axis=0)
        return preds
        #return preds
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_proba = self.predict_proba(X)

        pred = (pred_proba == np.expand_dims(pred_proba.max(axis=1), axis=1)).astype(int)

        return pred.ravel()
    
    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False
    def get_params(self, deep=True):
        dictionary = {"model":self.model}
        return dictionary



def get_model_quantile(input_shape, hidden_units=[64, 64]):
    """
    Creates and compiles a neural network model for quantile regression.
    
    Parameters:
    - input_shape: tuple, the shape of the input data (number of features,)
    - hidden_units: list, the number of units in each hidden layer
    
    Returns:
    - model: compiled Keras model
    """
    input_layer = layers.Input(shape=input_shape)
    hidden_layer = input_layer

    for units in hidden_units:
        hidden_layer = layers.Dense(units, activation='relu')(hidden_layer)

    output_10 = layers.Dense(1, name='quantile_10')(hidden_layer)
    output_50 = layers.Dense(1, name='quantile_50')(hidden_layer)
    output_90 = layers.Dense(1, name='quantile_90')(hidden_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=[output_10, output_50, output_90])

    model.compile(optimizer='adam',
                  loss={'quantile_10': quantile_loss_10,
                        'quantile_50': quantile_loss_50,
                        'quantile_90': quantile_loss_90})

    return model

# Example usage:
# model = get_model(input_shape=(n_features,))
# model.fit(X_train, [y_train, y_train, y_train], epochs=100, batch_size=32)

def get_model(input_shape):
    """
    Creates and compiles a neural network model for regression problems.
    
    Parameters:
    input_shape (int or tuple): The shape of the input data. For example, (10,) for 10 features.
    
    Returns:
    model (tf.keras.Model): The compiled regression model.
    """
    
    model = Sequential()
    
    # Input layer and first hidden layer
    model.add(Dense(64, activation='linear', input_shape=input_shape))
    
    # Second hidden layer
    model.add(Dense(32, activation='linear'))
    
    # Third hidden layer
    model.add(Dense(16, activation='linear'))
    
    # Output layer for regression (single continuous value)
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

num_features = 1
tf_model = get_model_quantile((num_features,))
tf_model = get_model((num_features,))
tf_model.summary()
tf_wrapped_model = TensorflowToMapie(tf_model)

x_train = {}
y_train = {}
x_test = {}
y_test = {}

x_train["sinex_het"], y_train["sinex_het"] = gen.generate_data(1000, [0.1],"normal","sinex_het")
x_test["sinex_het"], y_test["sinex_het"] = gen.generate_data(1000, [0.1],"normal","sinex_het")


mapie_model = MapieRegressor(estimator=tf_wrapped_model, method="plus", cv="split", random_state=42) 
mapie_model.fit(x_train["sinex_het"][0.1].ravel(), y_train["sinex_het"][0.1].ravel())
#%%
y_preds, y_pss = mapie_model.predict(x_test["sinex_het"][0.1].ravel(), alpha=0.1)

#%%
print(y_preds.ravel())
print(y_pss.ravel())
print(x_train["sinex_het"][0.1].ravel())