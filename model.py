import sys
import tensorflow as tf
import joblib
import numpy as np
import scipy
from tensorflow import keras
from tensorflow.keras import layers

def genhurst(S, q):
    num_examples = S.shape[0]
    epsilon = 1e-8
    L = S.shape[1]
    # if L < 100:
    #    warnings.warn('Data series very short!')

    H = np.zeros((num_examples, len(range(5, 20)), 1))
    k = 0

    for Tmax in range(5, 20):
        x = np.arange(1, Tmax + 1, 1)
        mcord = np.zeros((num_examples, Tmax, 1))

        for tt in range(1, Tmax + 1):

            dV = S[:, np.arange(tt, L, tt)] - S[:, np.arange(tt, L, tt) - tt]
            VV = S[:, np.arange(tt, L + tt, tt) - tt]
            N = dV.shape[1] + 1
            X = np.arange(1, N + 1, dtype=np.float64)
            Y = VV
            mx = np.sum(X) / N
            SSxx = np.sum(X**2) - N * mx**2
            my = np.sum(Y, axis=1) / N
            mul = np.multiply(X, Y)
            sum = np.sum(mul, axis=1)
            sum2 = np.sum(sum, axis=1, keepdims=True)
            SSxy = sum2 - N * mx * my
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            ddVd = dV - np.expand_dims(cc1, axis=2)
            VVVd = VV - np.expand_dims(np.multiply(cc1, np.arange(1, N + 1, dtype=np.float64)), axis=1) - np.expand_dims(cc2, axis=2)
            mcord[:, tt - 1] = np.expand_dims(np.mean(np.abs(ddVd) ** q, axis = (1, 2)) / (np.mean(np.abs(VVVd) ** q, axis = (1, 2)) + epsilon), axis=1)

        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx**2
        my = np.mean(np.log10(mcord + epsilon), axis=1)
        log_mcord = np.log10(mcord + epsilon)
        log_mcord_t = np.swapaxes(log_mcord, 1, 2)
        logx = np.log10(x)
        logx_2d = np.expand_dims(logx, axis=0)
        logx_3d = np.expand_dims(logx_2d, axis=0)
        mul_log_t = np.multiply(logx_3d, log_mcord_t)
        sum_log = np.sum(mul_log_t , axis=1)
        sum2_log = np.sum(sum_log, axis=1, keepdims=True)
        SSxy = sum2_log - Tmax * mx * my
        H[:, k] = SSxy / SSxx
        k = k + 1

    mH = np.mean(H, axis=1) / q
    mH = mH[:, 0]

    return mH

def calculate_ghe_overlap(time_series, len_win, q_max, step):
"""
Calculate the GHE matrix that will be the input of the attention mechanism
time_series is the time series that will be the input to the model (used to train it)
len_win is the size of the windows on which the GHE will be calculated. Recommended to be above 100
q_max the maximum value of q
step: the difference between the start of window n and the start of window n-1
"""
time_series_size = time_series.shape[1]
train_size = time_series.shape[0]
number_windows = (time_series_size - len_win) // step + 1
ghe_matrix = np.zeros((train_size, number_windows, q_max))

q = 1  # q do GHE
c = 0  # variável contadora do índice da série que estamos
i = 0  # variável para a armazenagem dos resultados

while q <= q_max:
    c = 0
    i = 0
    while (c + len_win) <= time_series_size:
        ghe_matrix[:,i, q - 1] = genhurst_vect(time_series[:, c : (c + len_win)], q)
        c = c + step
        i = i + 1
    q = q + 1

return ghe_matrix

def dot_scaled_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)


    return output, attention_weights

class mine_MultiHeadAttention(tf.keras.layers.Layer):
    # input = (batch_size, num_windows, num_expoents)
    def __init__(self, d_model, n_heads):
        super(mine_MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        assert self.d_model % self.n_heads == 0

        self.depth = self.d_model // self.n_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        # shape = (batch_size, num_windows, num_expoents)
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        # shape = (batch_size, num_windows, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))

        # shape = (batch_size, num_heads, num_windows, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # shape = (batch_size, num_heads, num_windows, depth)
        scaled_attention, attention_weights = dot_scaled_product_attention(q, k, v)

        # shape = (batch_size, num_heads, num_windows, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # shape = (batch_size, num_windows, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # shape  = (batch_size, num_windows, d_model)

        output = self.dense(concat_attention)
        # shape  = (batch_size, num_windows, d_model)
        return output

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        n_classes: int,
        n_neurons: int,
        n_layers: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.n_classes = n_classes
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of fully connected layers each has ReLU activation
        self.hidden = [
            tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)
        ]
        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(
            theta_size, activation="linear", name="theta"
        )

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, : self.input_size], theta[:, -self.n_classes :]
        return backcast, forecast

class PFGB_model(keras.Model):
    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 n_classes: int,
                 n_neurons: int,
                 n_layers: int,
                 n_stacks: int,
                 n_expoents: int,
                 n_windows: int,
                 d_model: int,
                 n_heads: int,
                **kwargs):

        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.n_classes = n_classes
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.n_expoents = n_expoents
        self.n_windows = n_windows
        self.d_model = d_model
        self.n_heads = n_heads

        self.n_beats_block_layer = NBeatsBlock(input_size= self.input_size,
                theta_size=self.theta_size,
                n_classes= self.n_classes,
                n_neurons= self.n_neurons,
                n_layers= self.n_layers)

        self.mha_layer = mine_MultiHeadAttention(d_model = self.d_model, n_heads = self.n_heads)

        self.hidden_input = tf.keras.layers.Dense(self.input_size, activation="linear")

        self.fc_layer = tf.keras.layers.Dense(240, activation = "relu")

        self.fc_layer2 = tf.keras.layers.Dense(480, activation = "relu")

        self.softmax_layer = tf.keras.layers.Dense(self.n_classes, activation = "softmax")

        self.flatten_layer = tf.keras.layers.Flatten()

        self.classification_layer =  tf.keras.layers.Dense(self.n_classes, activation = "sigmoid")



    def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            # split the input data between the time series and the exponential attention results
            input_data_series, ghe_matrix_2d = inputs[:, :self.input_size], inputs[:, self.input_size:]
            ghe_matrix_3d = tf.reshape(ghe_matrix_2d, (batch_size, self.n_windows, self.n_expoents))
            # multi-head self-attention mechanism layer
            mha_1 = self.mha_layer(v=ghe_matrix_3d, k=ghe_matrix_3d, q=ghe_matrix_3d)
            mha_1 = tf.reshape(mha_1, (batch_size, self.n_windows, self.d_model))


            flatten_layer = self.flatten_layer(mha_1)
            mha_nbeats_input = self.hidden_input(flatten_layer)
            concat_layer = keras.layers.concatenate([input_data_series, mha_nbeats_input])
            backcast, forecast = self.n_beats_block_layer(concat_layer)

            residuals = layers.subtract([input_data_series, backcast])

            for i, _ in enumerate(range(self.n_stacks-1)):

                concat_layer = keras.layers.concatenate([residuals, mha_nbeats_input])
                backcast, block_forecast = self.n_beats_block_layer(concat_layer)
                residuals = layers.subtract([residuals, backcast])
                forecast = layers.add([forecast, block_forecast])


            forecast = self.fc_layer(forecast)
            forecast = self.fc_layer2(forecast)
            final_forecast = self.classification_layer(forecast)

            return final_forecast
