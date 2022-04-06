"""A custom Attention layer with indoor localization in mind
"""

# supress warning msgs
# I know I don not have a GPU!
# https://stackoverflow.com/a/42121886
#yapf: disable
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#yapf: enable

import numpy as np
import tensorflow as tf
from typing import List


class ParisAttention(tf.keras.layers.Layer):
    """
    A variation on the implementation of Attention layer
    see: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

    key difference:
    If the train_values is not provided, return attention vector only

    """
    def __init__(self,
                 train_keys: np.array,
                 train_values=None,
                 trainable=False,
                 name="FingerprintAttention",
                 dtype=None,
                 dynamic=False,
                 outshape=None,
                 return_images=True,
                 **kwargs):
        """
        train_keys: np.array
            flattened input RSSI
        train_values: np.array
            flat integer labels, Default is None
        outshape: tuple
            shape of out put WITHOUT channel
        return_images: bool
            If true, return output as (N, N, 1)
            This is helpful if concat with convolution layer
        """
        # call the super class
        super(ParisAttention, self).__init__(trainable=trainable,
                                        name=name,
                                        dtype=dtype,
                                        dynamic=dynamic,
                                        **kwargs)

        # ######################################
        # save the train keys i.e. fingerprints
        self.train_keys = train_keys
        self.return_images = return_images
        # #######################################

        # #######################################
        # train values are generally target labels
        # if train_values is not one hot encoded; do so
        if train_values is not None and len(train_values.shape) == 1:
            train_values = tf.one_hot(train_values, len(tf.unique(train_values)[0]))
        self.train_values = train_values
        # #######################################

        # #######################################
        # compute padding for output; if required
        if self.return_images:
            # if train_values are provided
            # then compute padding based on it
            if self.train_values is not None:
                self.num_pads = self.compute_num_pads(
                    self.train_values.shape[1],
                    force_shape=outshape,
                )
                img_side = tf.sqrt(self.train_values.shape[1] + tf.cast(self.num_pads, tf.float32))
            # if train values are not provided then
            # compute padding based on train_keys
            else:
                self.num_pads = self.compute_num_pads(
                    self.train_keys.shape[0],
                    force_shape=outshape,
                )
                img_side = int(
                    tf.sqrt(self.train_keys.shape[0] + tf.cast(self.num_pads, tf.float32)))

            self.img_shape = (img_side, img_side, 1)
        # #######################################

    def call(self, inputs, *args, **kwargs):
        """Computes the actual attention and weighted train values

        Parameters
        ----------
        inputs : np.array or tf.Tensor
            Querty to the attention layer

        Returns
        -------
        tf.Tensor
            Attention or Weighted values based on layer config
        """

        # do calls asscoated with superclass
        super(ParisAttention, self).call(inputs, *args, **kwargs)

        # attention
        # one columns for each row of train_keys
        dots = tf.matmul(inputs, self.train_keys, transpose_b=True)
        distribution = tf.nn.softmax(dots)

        output = None

        if self.train_values is None:
            output = distribution
        else:
            # same shape as train_values
            output = tf.matmul(distribution, self.train_values)

        # app the output format if required
        return self.apply_output_format(output)

    def apply_output_format(self, x: tf.Tensor):
        """Apply the expected output format
        which is either flat vectors or images

        Parameters
        ----------
        x : tf.Tensor
            vectors with shape BS x number of columns

        Returns
        -------
        tf.Tensor
            [description]
        """

        # do nothing if images are not expected
        if not self.return_images:
            return x

        # create the padding required
        pad = tf.zeros((tf.shape(x)[0], self.num_pads), dtype=x.dtype)
        # concatenate the pad column wise
        # now the each vector length is a perfect square
        img_able = tf.concat((x, pad), 1)

        images = tf.reshape(img_able, (-1, *self.img_shape))

        return images

    @staticmethod
    def compute_num_pads(x: int, force_shape=None):
        """Compute the number of elements required
        such that x becomes a perfect square

        Parameters
        ----------
        x : int
            An integer representing the number of columns in a 2D array

        Returns
        -------
        tf.Tensor
            A tensor with a single value
        """
        if force_shape is not None:
            upper_sqrt = force_shape[0]
        else:
            upper_sqrt = tf.math.ceil(tf.math.sqrt(float(x)))

        padding = tf.cast((upper_sqrt * upper_sqrt) - float(x), tf.int32)

        return padding


class ParisMultiHeadAttention(tf.keras.layers.Layer):
    """
    An Attention layer with several Fingerprint Attention Layers inside
    """
    def __init__(
        self,
        attention_layers: List[ParisAttention],
        name="",
        **kwargs,
    ) -> None:
        """Init method for MultiHeadAttention

        Parameters
        ----------
        attention_layers : List[FingerprintAttention]
            list of pre-initialized attention layers
            the expected output shape of all layers is in same
        name : str, optional
            name of this layer, by default ""
        """
        # call the super init method
        super(ParisMultiHeadAttention, self).__init__(
            trainable=False,
            name=name,
            **kwargs,
        )

        # Can directly manipulate this list to add or remove
        # save the layers, they will be usefull
        self.attention_layers = attention_layers

        # check if all layers have some output shape
        self.img_shape = attention_layers[0].img_shape
        if len(attention_layers) > 0:
            for attn_layer in attention_layers[1:]:
                assert self.img_shape == attn_layer.img_shape, "shape mismatch"

    def call(self, inputs, *args, **kwargs):
        """Execute the single headed attention layers

        Parameters
        ----------
        inputs : tf.Tensor like
            Inputs to this layer
        """

        # list to store outputs
        outputs = []

        # execute each attention layer
        for attention_layer in self.attention_layers:
            attn = attention_layer(inputs)

            # attn is of shape (BS, N, N, 1)
            # store the outputs
            outputs.append(attn)

        # concatenate the outputs on the last axis
        # this way we do not have to worry about shapes in this layer
        new_shape = (-1, *self.img_shape[:2], len(self.attention_layers))
        return tf.reshape(tf.concat([outputs], axis=-1), new_shape)


# ##############################################################################
# Test And Maintainence Code
# ##############################################################################


def static_test_demo():
    test_query = np.array([
        [0.2, 0.1, 0.1],
        [0.2, 0.7, 0.2],
    ])  # 1x3, BS x Tq x 1 <= in TF lingo

    train_keys = np.array([
        [0.2, 0.1, 0.1],
        [0.6, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        # [0.2, 0.7, 0.1],
    ])  # 4x3

    train_values = np.array([1, 2, 3])  # 3x3

    # expect three attention values
    fa = ParisAttention(
        train_keys,
        train_values=train_values,
        outshape=(3, 3),
    )

    fa(test_query)

    # # output will be

    # expected = tf.Variable([
    #     [0.3168557, 0.34669536, 0.33644897],
    #     [0.27145913, 0.3153905, 0.41315034],
    # ])
    # if tf.reduce_all(tf.math.equal(fa(test_query), expected)):
    #     print("Static Test Pass")
    # else:
    #     print("Static Test Fail")


def test_static_multihead():

    test_query = np.array([
        [0.2, 0.1, 0.1],
        # [0.2, 0.7, 0.2],
    ])  # 1x3, BS x Tq x 1 <= in TF lingo

    train_keys0 = np.array([
        [0.2, 0.1, 0.1],
        [0.6, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        # [0.2, 0.7, 0.1],
    ])

    train_keys1 = np.array([
        [0.6, 0.2, 0.1],
        [0.2, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        # [0.2, 0.7, 0.1],
    ])

    train_keys2 = np.array([
        [0.2, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.2, 0.7, 0.1],
        # [0.6, 0.2, 0.1],
    ])

    # make 3 attention layers
    attention_layers = []
    for tk in [train_keys0, train_keys1, train_keys2]:
        fa = ParisAttention(tk)
        attention_layers.append(fa)

    # make multihead attention layer
    ma = ParisMultiHeadAttention(attention_layers)

    out = ma(test_query)

    print(out)

    print(tf.keras.layers.Conv2D(50, 2)(out))


def model_demo():

    DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    path = tf.keras.utils.get_file("mnist.npz", DATA_URL)
    with np.load(path) as data:
        train_x = data["x_train"]
        train_y = data["y_train"]
        test_x = data["x_test"]
        test_y = data["y_test"]

    img_shape = (28, 28, 1)
    num_samples = train_x.shape[0]

    kx = train_x.reshape((num_samples, img_shape[0] * img_shape[1]))

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=img_shape),
        tf.keras.layers.Flatten(),
        # FingerprintAttention(kx),
        # tf.keras.layers.Reshape((10, 10, 1)), # output is num train_samples
        # tf.keras.layers.Conv2D(50, 2, activation='relu'),
        # tf.keras.layers.Conv2D(50, 2, activation='relu'),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.summary()

    model.compile(
        optimizer="Adam",
        metrics=["Accuracy"],
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )

    model.fit(
        train_x,
        train_y,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
        validation_split=0.2,
    )

    print(model.evaluate(test_x, test_y))


if __name__ == "__main__":

    # model_demo()

    test_static_multihead()

    exit()
    static_test_demo()

    exit()

    tq = np.array([
        [0.2, 0.1, 0.1],
        [0.2, 0.7, 0.2],
        [0.2, 0.7, 0.2],
        [0.2, 0.7, 0.2],
        [0.2, 0.7, 0.2],
    ])  # 1x3, BS x Tq x 1 <= in TF lingo

    tq = tf.Variable(tq)

    upper_sqrt = tf.math.ceil(tf.math.sqrt(float(tq.shape[1])))

    padding = (upper_sqrt * upper_sqrt) - float(tq.shape[1])

    pad = tf.zeros((tq.shape[0], padding), dtype=tq.dtype)

    print(tq)
    print(pad)

    img_able = tf.concat((tq, pad), 1)

    print(tf.reshape(img_able, (-1, upper_sqrt, upper_sqrt, 1)))
