"""
functions to quickly train and test localization techniques
"""
from numpy.lib.twodim_base import tri
from helpers import get_visible_waps, make_images, split_frame
from Seth import Floorplan
from Seth import MAC_RE, Devices, fetch_seth
from stone import (
    TripletManager as StoneTripletManager,
    embedding_model as stone_embedding_model,
    complete_model as stone_complete_model,
    pick_negative_1d as stone_pick_negative_1d,
    attn_embedding_model as attn_embedding_model,
    attn_complete_model as attn_complete_model
)
from stone.siamese import load_encoder as stone_load_encoder
# from paris import (
#     TripletManager as ParisTripletManager,
#     embedding_model as paris_embedding_model,
#     complete_model as paris_complete_model,
#     pick_negative_1d as paris_pick_negative_1d,
# )
# from paris.siamese import load_encoder, save_encoder
# from paris.attention import (
#     ParisAttention as ParisAttention,
#     ParisMultiHeadAttention as ParisMultiHeadAttention,
# )
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.callbacks import TQDMProgressBar
from lt import LTKNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from data_helper import get_aps_generic as get_aps
from lt import LTKNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from data_helper import get_aps_generic as get_aps
from tensorflow.keras.models import load_model
import pandas as pd
from typing import List
import tensorflow as tf
from tensorflow_addons.callbacks import TQDMProgressBar




# ######################################
# KNN
# ######################################
def knn_train(train_df, target=["label"], nn=3):
    # get 1st data
    df = train_df

    train_waps = get_aps(train_df.columns)

    # target label
    true_y = df[target].values.flatten()

    # train model
    model = KNN(n_neighbors=nn)

    model.fit(np.array(df[train_waps].values), np.array(true_y, dtype=np.int))

    return model, train_waps


def knn_predict(model, test_df, train_waps):

    test_waps = get_aps(test_df.columns)

    missing_waps = list(set(train_waps) - set(test_waps))
    test_df[missing_waps] = -100

    test_x = test_df[train_waps]

    return model.predict(test_x)


# #####################################


# ######################################
# LT-KNN
# ######################################
def lt_knn_train(train_df, target=["label"], nn=3):

    train_waps = get_aps(train_df.columns)

    model = LTKNN(train_df[[*train_waps, *target]], train_waps, target, nn=nn)

    return model


def lt_knn_predict(model, test_df):

    train_waps = model.original_waps

    # update the regression model
    # test_waps are waps visible in online phase
    visible_waps = get_visible_waps(test_df, wap_re=MAC_RE)
    model.update(visible_waps)

    # setup test data
    test_x = test_df[train_waps]

    return model.predict(test_x)


# #####################################


# ######################################
# Stone
# ######################################
def stone_train(train_df,
                val_df=None,
                target=["label"],
                dim_embed=3,
                input_shape=(18, 18, 1),
                learning_rate=1e-4,
                alpha=0.5,
                batch_size=32,
                steps_per_epoch=100,
                p_turn_off=0.50,
                contrast_range=None,
                brightness_delta=None,
                gaussian_noise=0.10,
                model_layers=[50, 50, 100],
                epochs=100,
                callback_loss_patience=20,
                fit_verbose=0,
                nn=1,
                val_bs=100,
                encoder_path=None):

    if encoder_path is not None:
        triplet_encoder, meta = stone_load_encoder(encoder_path)
        train_waps = list(meta["TRAIN_WAPS"])
    else:
        # get train aps
        train_waps = get_aps(train_df.columns)

    if val_df is None:
        train_df, val_df = split_frame(train_df)

    # set up train data
    train_x = (train_df[train_waps].values + 100) / 100
    train_x = make_images(train_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    train_y = train_df[target].values.reshape((-1)).astype(int)

    # setup val data
    # set up train data
    val_x = (val_df[train_waps].values + 100) / 100
    val_x = make_images(val_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    val_y = val_df[target].values.reshape((-1)).astype(int)

    if encoder_path is None:

        triplet_encoder = stone_embedding_model(input_shape,
                                                dim_embed,
                                                gaussian_noise=gaussian_noise,
                                                model_layers=model_layers)

        # put the encoder into the stone system
        siamese = stone_complete_model(triplet_encoder, input_shape, learning_rate, alpha)
        
        # setup data generators and feed the monster!
        train_gen = StoneTripletManager(train_x,
                                        train_y,
                                        n_sampler=stone_pick_negative_1d,
                                        steps_per_epoch=steps_per_epoch,
                                        p_turn_off=p_turn_off,
                                        contrast_range=contrast_range,
                                        brightness_delta=brightness_delta,
                                        bs=batch_size)

        # validation data generator
        val_gen = StoneTripletManager(val_x,
                                      val_y,
                                      n_sampler=stone_pick_negative_1d,
                                      steps_per_epoch=steps_per_epoch,
                                      p_turn_off=p_turn_off,
                                      contrast_range=contrast_range,
                                      brightness_delta=brightness_delta,
                                      bs=val_bs)

        # fit the model
        history = siamese.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              verbose=fit_verbose,
                              callbacks=[
                                  EarlyStopping(monitor="val_loss",
                                                patience=callback_loss_patience,
                                                restore_best_weights=True),
                                  TQDMProgressBar(show_epoch_progress=False,
                                                  leave_overall_progress=False)
                              ])

        print("Loss:", history.history["loss"][-1])

    predictor = KNN(nn)
    # NOTE: Is it better to train using Augmented Data?
    train_encodings = triplet_encoder.predict(train_x)
    predictor.fit(train_encodings, train_y.flatten())

    return triplet_encoder, predictor, train_waps


def stone_test(test_df, encoder, predictor, train_waps, input_shape=(18, 18, 1)):

    # any waps mising can be fixed
    test_waps = get_aps(list(test_df.columns))
    missing_waps = set(train_waps) - set(test_waps)
    test_df[list(missing_waps)] = -100

    # tx
    tx = np.array(test_df[train_waps].values, dtype=np.float)
    tx = (tx + 100) / 100
    tx = make_images(tx, force_shape=input_shape[:2])

    # test_y = np.array(test_df["label"].values, dtype=np.int).flatten()
    # test_y = test_y.reshape((-1, 1))

    # predict test using "train_waps"
    encoded = encoder.predict(tx)
    return predictor.predict(encoded).flatten()

# Attention
def attention_train(train_df,
                val_df=None,
                target=["label"],
                dim_embed=3,
                input_shape=(18, 18, 1),
                learning_rate=1e-4,
                alpha=0.5,
                batch_size=32,
                steps_per_epoch=100,
                p_turn_off=0.50,
                contrast_range=None,
                brightness_delta=None,
                gaussian_noise=0.10,
                model_layers=[50, 50, 100],
                epochs=100,
                callback_loss_patience=20,
                fit_verbose=0,
                nn=1,
                val_bs=100,
                encoder_path=None):        # Attention

 
    train_waps = get_aps(train_df.columns)

    if val_df is None:
        train_df, val_df = split_frame(train_df)

    # set up train data
    train_x = (train_df[train_waps].values + 100) / 100
    train_x = make_images(train_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    train_y = train_df[target].values.reshape((-1)).astype(int)

    # setup val data
    # set up train data
    val_x = (val_df[train_waps].values + 100) / 100
    val_x = make_images(val_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    val_y = val_df[target].values.reshape((-1)).astype(int)

    if encoder_path is None:

        triplet_encoder = attn_embedding_model(input_shape,
                                                dim_embed,
                                                gaussian_noise=gaussian_noise,
                                                model_layers=model_layers)

        # put the encoder into the stone system
        attention = attn_complete_model(triplet_encoder, input_shape, learning_rate, alpha)
        
        # setup data generators and feed the monster!
        train_gen = StoneTripletManager(train_x,
                                        train_y,
                                        n_sampler=stone_pick_negative_1d,
                                        steps_per_epoch=steps_per_epoch,
                                        p_turn_off=p_turn_off,
                                        contrast_range=contrast_range,
                                        brightness_delta=brightness_delta,
                                        bs=batch_size)

        # validation data generator
        val_gen = StoneTripletManager(val_x,
                                      val_y,
                                      n_sampler=stone_pick_negative_1d,
                                      steps_per_epoch=steps_per_epoch,
                                      p_turn_off=p_turn_off,
                                      contrast_range=contrast_range,
                                      brightness_delta=brightness_delta,
                                      bs=val_bs)

        # fit the model
        history = attention.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              verbose=fit_verbose,
                              callbacks=[
                                  EarlyStopping(monitor="val_loss",
                                                patience=callback_loss_patience,
                                                restore_best_weights=True),
                                  TQDMProgressBar(show_epoch_progress=False,
                                                  leave_overall_progress=False)
                              ])

        print("Loss:", history.history["loss"][-1])

    predictor = KNN(nn)
    # NOTE: Is it better to train using Augmented Data?
    train_encodings = triplet_encoder.predict(train_x)
    predictor.fit(train_encodings, train_y.flatten())

    return triplet_encoder, predictor, train_waps
    # end
        


# #####################################

# ######################################
# Paris
# ######################################
from paris.ParisModel import ParisAttentionModel


def paris_train_model(
    device,
    floor,
    ci,
    model_save_path=None,
    attention_fetch_kwgs=None,
    **kwargs,
):

    floorplan_train_config = {
        Floorplan.BASEMENT: {
            "input_shape": (20, 20, 1),
            "val_bs": 1024,
            "epochs": 100,
            "batch_size": 128,
            "steps_per_epoch": 100,
            "nn": 1,
            "model_layers": [50, 50, 100],
            "p_turn_off": 0.90,
            "contrast_range": (0.8, 1.2),
            "brightness_delta": None,
            "learning_rate": 1e-3,
            "alpha": 0.50,
            "callback_loss_patience": 10,
            "fit_verbose": 0,
            "floor": Floorplan.BASEMENT,
            "train_device": device,
            "collection_instance": ci,
            "dim_embed": 3,
            "attention_kwgs": attention_fetch_kwgs,
        },
    }

    # overwrite build config with any passed kwargs
    build_config = {**floorplan_train_config[floor], **kwargs}

    # fetch the current data
    train_df, _ = fetch_seth(device, floor, ci)
    tdf, vdf = split_frame(train_df)

    # get train_waps
    train_waps = get_aps(tdf)

    # set up train data
    train_x = (tdf[train_waps].values + 100) / 100
    train_x = make_images(
        train_x.astype(float),
        force_shape=build_config["input_shape"][:2],
    )
    # set outputs
    train_y = tdf[["label"]].values.reshape((-1)).astype(int)

    # setup val data
    # set up train data
    val_x = (vdf[train_waps].values + 100) / 100
    val_x = make_images(
        val_x.astype(float),
        force_shape=build_config["input_shape"][:2],
    )
    # set outputs
    val_y = vdf["label"].values.reshape((-1)).astype(int)

    # work on attention tables/values
    attention_shape = (train_x.shape[0], np.prod(build_config["input_shape"]))
    attention_tables = [train_x.reshape(attention_shape)]
    if attention_fetch_kwgs is not None:

        for kwg in attention_fetch_kwgs:
            attn_df, _ = fetch_seth(**kwg)
            test_waps = get_aps(list(attn_df.columns))
            missing_waps = set(train_waps) - set(test_waps)
            attn_df[list(missing_waps)] = -100

            attn_x = make_images(
                attn_df[train_waps].astype(float),
                force_shape=build_config["input_shape"][:2],
            ).reshape((-1, attention_shape[1]))

            attention_tables.append(attn_x)

    #######################################################
    # build the encoder
    triplet_encoder = ParisAttentionModel(
        attention_tables,
        train_waps=train_waps,
        dim_embed=build_config["dim_embed"],
    )
    ######################################################

    siamese = paris_complete_model(
        triplet_encoder,
        build_config["input_shape"],
        lr=build_config["learning_rate"],
        alpha=build_config["alpha"],
    )

    # setup data generators and feed the monster!
    train_gen = ParisTripletManager(train_x,
                                    train_y,
                                    n_sampler=paris_pick_negative_1d,
                                    steps_per_epoch=build_config["steps_per_epoch"],
                                    p_turn_off=build_config["p_turn_off"],
                                    contrast_range=build_config["contrast_range"],
                                    brightness_delta=build_config["brightness_delta"],
                                    bs=build_config["batch_size"])

    # validation data generator
    val_gen = ParisTripletManager(val_x,
                                  val_y,
                                  n_sampler=paris_pick_negative_1d,
                                  steps_per_epoch=build_config["steps_per_epoch"],
                                  p_turn_off=build_config["p_turn_off"],
                                  contrast_range=build_config["contrast_range"],
                                  brightness_delta=build_config["brightness_delta"],
                                  bs=build_config["val_bs"])

    # fit the model
    siamese.fit(train_gen,
                          validation_data=val_gen,
                          epochs=build_config["epochs"],
                          verbose=build_config["fit_verbose"],
                          callbacks=[
                              tf.keras.callbacks.EarlyStopping(
                                  monitor="val_loss",
                                  patience=build_config["callback_loss_patience"],
                                  restore_best_weights=True),
                              TQDMProgressBar(show_epoch_progress=False,
                                              leave_overall_progress=False)
                          ])

    if model_save_path is not None:
        save_encoder(triplet_encoder, {
            "TRAIN_WAPS": train_waps,
            "BUILD_CONFIG": {
                **build_config
            }
        },
                                     save_path=model_save_path)

    predictor = paris_make_predictor(train_df, train_waps, triplet_encoder,
                                     build_config["input_shape"], build_config["nn"])

    return triplet_encoder, predictor, train_waps


def paris_make_predictor(test_df, train_waps, encoder, input_shape, nn=1):
    # any waps mising can be fixed
    test_waps = get_aps(list(test_df.columns))
    missing_waps = set(train_waps) - set(test_waps)
    test_df[list(missing_waps)] = -100

    # tx
    tx = np.array(test_df[train_waps].values, dtype=np.float)
    tx = (tx + 100) / 100
    tx = make_images(tx, force_shape=input_shape[:2])
    ty = test_df[["label"]].values.reshape((-1)).astype(int)

    # predict test using "train_waps"
    encoded = encoder.predict(tx)
    return KNN(nn).fit(encoded, ty), (encoded, tx, ty)


def paris_test(test_df, encoder, train_waps, predictor=None, input_shape=(20, 20, 1)):

    if predictor is None:
        predictor, (_, tx, _) = paris_make_predictor(test_df, train_waps, encoder, input_shape)

    # predict test using "train_waps"
    encoded = encoder.predict(tx)
    return predictor.predict(encoded).flatten()


# #####################################
