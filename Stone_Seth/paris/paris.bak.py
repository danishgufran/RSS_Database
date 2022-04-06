"""
Implement Paris as a class
Enables maintaing multiple variables over time
"""
from typing import List
import json
import pandas as pd
import numpy as np

try:
    try:
        from attention import (
            ParisAttention,
            ParisMultiHeadAttention,
        )
    except:
        from paris.attention import (
            ParisAttention,
            ParisMultiHeadAttention,
        )

except:
    from LifeLongLoc.paris.attention import (
        ParisAttention,
        ParisMultiHeadAttention,
    )

from tensorflow import keras
from tensorflow import concat as concat_layers


class ParisMultiHeadEncoder(keras.Model):
    def __init__(
        self,
        train_df,  # training fingerprints
        train_waps=None,  # train waps if provided
        input_shape: tuple = (20, 20, 1),  # input image shape; 
        triplet_encoder: keras.Model = None,  # encoder model including attention
        triplet_encoder_path: str = None,  # load triplet encoder model from a specific file
        max_heads: int = 20,  # maximum number of heads you can have
        dim_embed: int = 3,  # size of siamese embedding
        gaussian_noise: float = 0.10,  # input gaussian noise
        model_layers: List[int] = [50, 50, 100],  # model layers if no encoder model provided
        post_attention_filters=[50],  # layer that comes after MHA before joining main model
        attention_in_idx=0,  # num layer into which attention is injected
        # head_dim_out: int = (50, 50),  # max samples ; should be perfect sqaure
    ):

        # call init of superclass
        super(ParisMultiHeadEncoder, self).__init__()

        ############################################################
        # save main model params
        self.dim_embed = dim_embed
        self.gaussian_noise = gaussian_noise
        self.model_layers = model_layers
        self.post_attention_filters = post_attention_filters
        self.attention_in_idx = attention_in_idx
        ############################################################

        ############################################################
        # make list of train_waps
        if train_waps is None:
            self.train_waps = self.get_aps(train_df.columns)
        self.shape_input = input_shape
        ############################################################

        ############################################################
        # train matrices
        # train_x is in 2D shape
        self.tx_vecs = self.__make_attention_vectors__(train_df)
        ############################################################

        ##################################################################
        # fingerprints that we have observed so far
        # at init we only have training data fingerprints
        # list of numpy arrays
        self.collected_fingerprints = [self.tx_vecs]
        ##################################################################

        ##################################################################
        # Varibles for Multihead
        self.head_dim_out = self.shape_input[:2]  # head_dim_out
        self.max_heads = max_heads
        self.multi_head_attention_layer = None
        ##################################################################

        ##################################################################
        # load the encoder if provided
        # TODO: extract Multihead
        if triplet_encoder_path is not None:
            self.triplet_encoder, meta = self.load_encoder(triplet_encoder_path)
            self.train_waps = meta["TRAIN_WAPS"]
        elif triplet_encoder is not None:
            self.triplet_encoder = triplet_encoder
        else:
            self.triplet_encoder = None
            self.build_encoder()
            # TODO: if triplet encoder lodaded from file or externally
            # find and save pointer to the multi_head_attention_layer
            # maybe usefull later; but not really
        ##################################################################

    def call(self, inputs):
        return self.triplet_encoder(inputs)

    def update(self, dfs: List[pd.DataFrame], rebuild=True):
        # TODO:
        # process dataframes provided
        txs = []
        for frame in dfs:
            tx = self.__make_attention_vectors__(frame)
            txs.append(tx)
        # create one entry in collected_fingerprints
        txs = np.vstack(txs)
        self.collected_fingerprints.append(txs)

        # call __build_model__()
        if rebuild:
            self.build_encoder()

    def build_encoder(self):

        ##################################################################
        # basic input stuff
        input_layer = keras.Input(shape=self.shape_input)
        # add gaussian noise
        conn = keras.layers.GaussianNoise(self.gaussian_noise)(input_layer)
        ##################################################################

        ##################################################################
        # send the connection to the attention layer

        attn_conn = keras.layers.Flatten()(conn)
        self.multi_head_attention_layer = self.__get_multi_head_attention__()
        attn_conn = self.multi_head_attention_layer(attn_conn)
        # pass the attn_conn to the post_attention layer
        for paf in self.post_attention_filters:
            attn_conn = keras.layers.Conv2D(
                filters=paf,
                kernel_size=2,
                strides=1,
                activation='relu',
                padding="same",
            )(attn_conn)
            attn_conn = keras.layers.Dropout(0.1)(attn_conn)
        # we will add attn_conn at the right index
        ##################################################################

        ##################################################################
        # Iterate over layers and add attention at the right spot
        # add conv body layers
        padding = "same"  # head output and input shape are set to same thing
        for i, num_filter in enumerate(self.model_layers[:-1]):

            # concatenate attention here
            if i == self.attention_in_idx:
                # FIXME: Make sure these layers are concatable!
                conn = concat_layers([conn, attn_conn], -1)
                padding = "valid"

            # include convolution layers
            conn = keras.layers.Conv2D(
                num_filter,
                kernel_size=(2, 2),
                strides=1,
                padding=padding,
                activation='relu',
            )(conn)
            conn = keras.layers.Dropout(0.1)(conn)

        # flatten layers
        conn = keras.layers.Flatten()(conn)

        # flattened layers post conv
        conn = keras.layers.Dense(
            self.model_layers[-1],
            activation='relu',
        )(conn)
        conn = keras.layers.Dropout(0.2)(conn)

        # setup output layer
        output_layer = keras.layers.Dense(self.dim_embed)(conn)

        # build the model
        self.triplet_encoder = keras.Model(input_layer, output_layer)
        # assuming this function act
        self.build((0, *self.shape_input))

    def __make_attention_vectors__(self, df, missing_val=-100):
        # any waps mising can be fixed
        test_waps = self.get_aps(list(df.columns))
        missing_waps = list(set(self.train_waps) - set(test_waps))
        if len(missing_waps) != 0:
            df[list(missing_waps)] = missing_val

        # prepare train_x and train_y
        vectors = (df[self.train_waps].values + 100) / 100
        # pad the vectors so that they match the input shape
        # we assume that the input shape is always larger than vector cols
        num_pad_cols = np.prod(self.shape_input) - vectors.shape[1]
        padding = np.zeros((vectors.shape[0], num_pad_cols))

        return np.hstack((vectors, padding))

    def __get_multi_head_attention__(self) -> keras.layers.Layer:
        # for each array in collected fingerprints
        # build an attention layer
        attention_layers = []
        # look at the most recently observed fingerprints first
        for i, cfp in enumerate(reversed(self.collected_fingerprints)):

            # check if we have exceeded the max heads
            if i == self.max_heads:
                break

            # shuffle the samples in place
            np.random.shuffle(cfp)

            # keep only the max_dim_out attention rows
            # or less; excess is ignored
            # we assume a large number to accomodate everything
            # if we have excess data
            if np.prod(self.head_dim_out) < cfp.shape[0]:
                cfp = cfp[:np.prod(self.head_dim_out)]
                # only keep as much as we can accomodate

            # build attention layer
            attn = ParisAttention(cfp, outshape=self.head_dim_out)

            # store layer in list
            attention_layers.append(attn)

        # Make the actual MultiHead attention layer here
        return ParisMultiHeadAttention(attention_layers)

    # ######################### #
    # toolbox methods
    # ######################### #
    @staticmethod
    def get_aps(columns):
        macs = []
        for lbl in columns:
            if "WAP_" in lbl or ":" in lbl:
                macs.append(lbl)
        return macs

    @staticmethod
    def save_encoder(encoder: keras.Model, meta: dict, save_path: str):
        encoder.save(save_path)
        json.dump(meta, open(f"{save_path}/model_info.json", "w"))

    @staticmethod
    def load_encoder(save_path: str, compile=False):
        # TODO: this should be a class method
        encoder = keras.models.load_model(save_path, compile=False)
        meta = json.load(open(f"{save_path}/model_info.json", "r"))
        return encoder, meta


###### END OF CLASS #######


def test_me():
    """
    from within super directory
    exec
    python3 paris/paris.py
    """
    from seth import Devices, Floorplan, fetch_seth
    import tensorflow as tf
    from siamese import (
        TripletManager,
        complete_model,
        pick_negative_1d,
    )
    from helpers import split_frame, make_images
    from sklearn.neighbors import KNeighborsClassifier

    train_df, _ = fetch_seth(Devices.lg, Floorplan.BASEMENT, 0)

    tdf, vdf = split_frame(train_df)

    input_shape = (50, 50, 1)

    val_bs = 32
    epochs = 1
    batch_size = 32
    steps_per_epoch = 3
    nn = 1
    model_layers = [50, 50, 100]
    p_turn_off = 0.90
    learning_rate = 1e-3
    alpha = 0.50
    callback_loss_patience = 10
    validation_steps = 1

    triplet_encoder = ParisMultiHeadEncoder(
        tdf,  # dataframe shape should match input shape
        input_shape=input_shape,
        model_layers=model_layers)

    # set up train data
    train_x = (train_df[triplet_encoder.train_waps].values + 100) / 100
    train_x = make_images(
        train_x.astype(float),
        force_shape=input_shape[:2],
    )
    # set outputs
    train_y = train_df[["label"]].values.reshape((-1)).astype(int)

    # setup val data
    # set up train data
    val_x = (vdf[triplet_encoder.train_waps].values + 100) / 100
    val_x = make_images(val_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    val_y = vdf["label"].values.reshape((-1)).astype(int)

    siamese = complete_model(
        triplet_encoder,
        input_shape,
        lr=learning_rate,
        alpha=alpha,
    )

    # setup data generators and feed the monster!
    train_gen = TripletManager(train_x,
                               train_y,
                               n_sampler=pick_negative_1d,
                               steps_per_epoch=steps_per_epoch,
                               p_turn_off=p_turn_off,
                               contrast_range=(0.8, 1.2),
                               brightness_delta=None,
                               bs=batch_size)

    # validation data generator
    val_gen = TripletManager(val_x,
                             val_y,
                             n_sampler=pick_negative_1d,
                             steps_per_epoch=steps_per_epoch,
                             p_turn_off=p_turn_off,
                             contrast_range=(0.8, 1.2),
                             brightness_delta=None,
                             bs=val_bs)

    # tf.keras.utils.plot_model(triplet_encoder.triplet_encoder, "plots/model.png", show_shapes=True)

    # fit the model
    history = siamese.fit(train_gen,
                          validation_data=val_gen,
                          validation_steps=validation_steps,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[
                              tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                               patience=callback_loss_patience,
                                                               restore_best_weights=True),
                          ])

    print(history.history["loss"])
    predictor = KNeighborsClassifier(nn)
    # NOTE: Is it better to train using Augmented Data?
    train_encodings = triplet_encoder.predict(train_x)
    predictor.fit(train_encodings, train_y.flatten())

    triplet_encoder.update([
        fetch_seth(Devices.lg, Floorplan.BASEMENT, ci=0)[0],
        fetch_seth(Devices.s7, Floorplan.BASEMENT, ci=0)[0],
        fetch_seth(Devices.htc, Floorplan.BASEMENT, ci=0)[0],
    ])

    siamese = complete_model(
        triplet_encoder,
        input_shape,
        lr=learning_rate,
        alpha=alpha,
    )

    # fit the model
    history = siamese.fit(train_gen,
                          validation_data=val_gen,
                          validation_steps=validation_steps,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[
                              tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                               patience=callback_loss_patience,
                                                               restore_best_weights=True),
                          ])


if __name__ == "__main__":
    test_me()
    # model = keras.models
