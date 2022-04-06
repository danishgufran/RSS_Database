from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
# Attention
from tensorflow.keras.layers import Attention
def attn_embedding_model(
        input_shape: tuple,
        num_outputs: int,
        gaussian_noise=None,
        custom_layers=None,
        model_layers=[50, 50, 100],
        kernel_size=(2, 2),
        strides=1,
):

    # make input layer
    query = keras.Input(shape=input_shape)
    value = keras.Input(shape=input_shape)
    conn = query

    if custom_layers is None:

        # add gaussian noise if provided
        if gaussian_noise is not None:
            query = keras.layers.GaussianNoise(gaussian_noise)(query)

        token_embedding = layers.Embedding(input_dim=1000, output_dim=64)
        query_embeddings = token_embedding(query)
        value_embeddings = token_embedding(value)

        for i, num_filter in enumerate(model_layers[:-1]):

            # include convolution layers
            conn = layers.Conv2D(
                num_filter,
                kernel_size,
                strides=strides,
                activation='relu',
            )(conn)

        # flatten layers
        conn = keras.layers.Flatten()(conn)

        # flattened layers post conv
        conn = keras.layers.Dense(
            model_layers[-1],
            activation='relu',
        )(conn)
        conn = keras.layers.Dropout(0.2)(conn)

    else:
        for my_layer in custom_layers:
            conn = my_layer(conn)

    query = layers.Conv1D(filters=100, kernel_size=4, padding='same')
    query_encoding = query(query_embeddings)
    value_encoding = query(value_embeddings)
    query_attention_seq = layers.Attention()([query_encoding, value_encoding])
    query_encoding = layers.GlobalAveragePooling3D()(query_encoding)
    query_value_attention = layers.GlobalAveragePooling3D()(query_attention_seq)
    input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])


    # output layer
    output_layer = keras.layers.Dense(num_outputs)(conn)

    return keras.Model(input_layer, output_layer)

def triplet_loss(x, alpha=0.2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
    return loss

def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def attn_complete_model(base_model, input_shape, lr, alpha):
    # Create the complete model with three
    # embedding models and minimize the loss
    # between their output embeddings
    input_1 = keras.layers.Input(input_shape)
    input_2 = keras.layers.Input(input_shape)
    input_3 = keras.layers.Input(input_shape)

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    def trip_loss(x):
        return triplet_loss(x, alpha)

    # ooh magic!
    loss = Attention(trip_loss)([A, P, N])
    model = keras.Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=tf.keras.optimizers.Adam(lr))
    return model