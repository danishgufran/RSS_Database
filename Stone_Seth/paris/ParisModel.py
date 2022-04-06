"""

A better implementation of the Paris Model

with NO custom layers

"""

import enum
from tensorflow import keras
from typing import List, Tuple
import tensorflow as tf
import logging as lg

class ParisMultiHeadAttention(keras.layers.Layer):
    """A custom MultiHead attention
    Mainly a collection of attention layers with output as images
    """
    def __init__(self, attention_values: List[tf.constant], **kwargs) -> None:
        """Contructor for ParisMultiHeadAttention

        Parameters
        ----------
        attention_values : List[tf.constant]
            2D arrays of fingerprints; num columns are expected to be perfect square
        """
        super(ParisMultiHeadAttention, self).__init__(**kwargs)

        # force float 32 to save some space
        # Only need two significant digits
        self.attention_values = [tf.constant(x, dtype=tf.float16) for x in attention_values]

        # compute outshape
        num_cols = tf.cast(tf.shape(self.attention_values[0])[1], tf.float32)
        s = tf.cast(tf.math.sqrt(num_cols), tf.int32)
        self.outshape = (s, s, 1)

        # attention layers
        self.attention_layers = []
        for i in range(len(self.attention_values)):
            attn = keras.layers.Attention(name=f"atten_{i}")
            self.attention_layers.append(attn)


    def call(self, inputs):
        """
        Call method on this layer
        inputs should be 2d batch
        """
        attentions = []
        # this executes for every batch
        for i, value in enumerate(self.attention_values):
            attn = self.attention_layers[i]([inputs, value])
            attn = tf.reshape(attn, (-1, *self.outshape))
            attentions.append(attn)
        # return while concat on channels
        attentions = tf.concat(attentions, axis=-1)

        return attentions

class ParisAttentionModel(keras.Model):
    # TODO: implement get_config if want to use Cusotm Attenton Layer
    def __init__(
            self,
            attention_values: List[tf.constant],  # 2d numpy array
            train_waps=None,  # wifi APs associated with this model;
            dim_embed: int = 3,  # size of siamese embedding
            name="ParisMultiHeadEncoder",
            **kwargs):
        # pass the input shape to super class
        super(ParisAttentionModel, self).__init__(name=name, **kwargs)

        # ####################################################
        # store all the properties in self variables
        #
        if train_waps:
            self.train_waps = tf.constant(train_waps, dtype=tf.string)
        else:
            self.train_waps = tf.constant([], dtype=tf.string)
        self.dim_embed = dim_embed
        # ####################################################

        # ####################################################
        # Main Model layers
        #
        # input block 0
        self.gaussian_noise = keras.layers.GaussianNoise(0.1)
        self.conv2d_0 = keras.layers.Conv2D(
            kwargs.get("conv2d_0_num_filters", 50),
            kernel_size=kwargs.get("kernel_size", (2, 2)),
            strides=kwargs.get("strides", 1),
            activation='relu',
            name=f"Conv2D_0",
            padding="same",
        )
        self.dropout_0 = keras.layers.Dropout(0.1)

        # block 1
        self.conv2d_1 = keras.layers.Conv2D(
            kwargs.get("conv2d_1_num_filters", 50),
            kernel_size=kwargs.get("kernel_size", (2, 2)),
            strides=kwargs.get("strides", 1),
            activation='relu',
            name=f"Conv2D_1",
            padding="valid",
        )
        self.dropout_1 = keras.layers.Dropout(0.1)

        # flatten
        self.flatten = keras.layers.Flatten()

        # dense layers
        self.dense_0 = keras.layers.Dense(
            kwargs.get("dense_0_num_units", 100),
            activation='relu',
        )
        self.dropout_2 = keras.layers.Dropout(0.1)

        # embedding layer with no activation
        self.embedding = keras.layers.Dense(self.dim_embed)
        # ####################################################

        # ####################################################
        # Create attention branch layers and variables
        #
        # Store attention layer "V"s as a list of constants
        self.pre_attn_flatten = keras.layers.Flatten()
        self.multi_head_attention = ParisMultiHeadAttention(attention_values)

        self.attn_conv = keras.layers.Conv2D(
            kwargs.get("attn_conv_num_filters", 50),
            kernel_size=kwargs.get("attn_kernel_size", (2, 2)),
            strides=kwargs.get("attn_strides", 1),
            activation='relu',
            name=f"Attn_Conv2D",
            padding="same",
        )
        self.attn_dropout = keras.layers.Dropout(0.1)
        # ####################################################

    def call(self, inputs):
        """Call method of model

        General model description

              Attention --> Conv2D --|
        Input   ^                    V
          |--> GN ------------->  Conv2D --> Conv2D --> Dense --> Embedding

        Parameters
        ----------
        inputs : tf.Tensor or similar
            input to model
        """
        # apply gaussian noise
        gn = self.gaussian_noise(inputs)

        # main branch
        # block 0 --> padding is same
        x = self.conv2d_0(gn)
        x = self.dropout_0(x)

        # attention block in parallel to main branch
        attn_x = self.pre_attn_flatten(gn)
        attn_x = self.multi_head_attention(attn_x)
        attn_x = self.attn_conv(attn_x)
        self.attn_dropout(attn_x)
        # concat attention with input x
        x = tf.concat([x, attn_x], axis=-1)


        # block 1
        x = self.conv2d_1(x)
        x = self.dropout_1(x)

        # flatten
        x = self.flatten(x)

        # dense block (bottleneck)
        x = self.dense_0(x)
        x = self.dropout_2(x)

        # embedding
        return self.embedding(x)

    def model(self, input_shape=(20, 20, 1)):
        x = keras.layers.Input(input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

    @tf.function(input_signature=[])
    def get_train_waps(self):
        return self.train_waps

# FIXME: The above model does not fit well
if __name__ == "__main__":

    import numpy as np

    x = np.random.random((5, 20, 20, 1))
    y = np.random.randint(0, 3, size=(5, ))

    # mha = ParisMultiHeadAttention([x.reshape((5, 20 * 20)), x.reshape((5, 20 * 20))])

    model = ParisAttentionModel(
        attention_values=[x.reshape((5, 20 * 20)), x.reshape((5, 20 * 20))],
        train_waps=["A", "B", "C"],
    )

    model.build(x.shape)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(x, y, steps_per_epoch=1, epochs=1)

    # model.save("dummy")

    # model.summary()

    # model.model().summary()

    tf.keras.utils.plot_model(model.model((20, 20, 1)), "plots/model.png", show_shapes=True)

    # keras.backend.clear_session()