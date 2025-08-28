import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


def init_model(X):

    # Shapes d'exemple — adapte C (canaux board) et S (nb de scalaires)
    C = 12     #(nb piece en tout donc 6 pièces de chaque coté)
    S = X[1]
    LR = 1e-3

    # === Inputs ===
    board_in = layers.Input(shape=(8, 8, C), name="board")        # si tu n'utilises pas le board, supprime cette branche
    scal_in  = layers.Input(shape=(S,),       name="scalars")

    # === Petite CNN (board) ===
    x = layers.Conv2D(32, 3, padding='same',activation="relu")(board_in)
    x = layers.Conv2D(32, 3, padding='same',activation="relu")(x)
    x = layers.Conv2D(64, 1, activation="relu")(x)
    flatten = tf.keras.layers.Flatten()(x)

    # === Petite MLP (scalars) ===
    s = layers.LayerNormalization()(scal_in)
    s = layers.Dense(64, activation="relu")(s)
    s = layers.Dense(64, activation="relu")(s)

    # === Fusion + tête régression bornée ===
    h = layers.Concatenate()([flatten, s])
    h = layers.Dense(64, activation="relu")(h)
    out = layers.Dense(1, activation="relu")(h)

    model = Model(inputs=[board_in, scal_in], outputs=out)
    # –– Compilation
    model.compile(
        loss="mse",
        optimizer=optimizers.Adam(learning_rate=LR),
        metrics=["mae"]
    )
    return model


BATCH = 64
EPOCHS = 4
patience = 5

def start_model(model, df_CNN_array, df_tabular, y, patience = patience, BATCH = BATCH, EPOCHS = EPOCHS):


    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="mae", patience, restore_best_weights=True)
    ]
    history = model.fit(
    {"board": df_CNN_array, "scalars": df_tabular}, y,
    batch_size=BATCH, epochs=EPOCHS, callbacks=callbacks, verbose=1
    )
    return history
