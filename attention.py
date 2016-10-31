#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Recurrent, Dense
from keras.engine import InputSpec
from keras import initializations, activations, regularizers
from keras import backend as K

class Attention(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

    # Notes
    z - conext vector
    a bold - annotation vectors i= 1..L corresponding to image features as location i
    a - positive weight: probability that location i is the right place to focus
    s_t,i - indicator one hot, t-word, i-location
    '''
    def __init__(self, output_dim,
                 attention='stochastic',
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,
                 semi_sampling_p=0.5, temperature=1.,
                 **kwargs):
        # self.h_init = Dense  #(output_dim, input_shape=input_shape[:-1])
        # self.c_init = Dense  #(output_dim, input_shape=input_shape[:-1])
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.attention = attention
        self.semi_sampling_p = semi_sampling_p
        self.temperature = temperature

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        self.states = [None, None]

        self.W_i = self.init((self.input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((self.input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((self.input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((self.input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        # weights for initializing hidden and cell states of the LSTM
        self.Hw_h = self.init((self.input_dim, self.output_dim), name='{}_Hw_h'.format(self.name))
        self.Hw_c = self.init((self.input_dim, self.output_dim), name='{}_Hw_c'.format(self.name))
        self.Hb_h = K.zeros((self.output_dim,), name='{}_Hb_h'.format(self.name))
        self.Hb_c = K.zeros((self.output_dim,), name='{}_Hb_c'.format(self.name))

        self.trainable_weights += [self.Hw_h, self.Hw_c,
                                   self.Hb_h, self.Hb_c]

        # attention specific weights
        self.Aw = self.init((self.input_dim + self.output_dim, 1), name='{}_Aw'.format(self.name))
        self.Ab = K.zeros((1,), name='{}_Ab'.format(self.name))

        self.trainable_weights += [self.Aw, self.Ab]


        self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
        self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
        self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, x):
        xmean = K.mean(x, axis=1)
        # input_shape = self.input_spec[0].shape
        # h0 = self.c_init(xmean)
        # c0 = self.h_init(xmean)
        h0 = self.inner_activation(K.dot(xmean, self.Hw_h) + self.Hb_h)
        c0 = self.inner_activation(K.dot(xmean, self.Hw_c) + self.Hb_c)
        return [h0, c0]

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        return x

    def step(self, x, states_constants):
        # TODO need to figure out where states comes from
        h_tm1 = states_constants[0]
        c_tm1 = states_constants[1]
        # x_tm1 = states_constants[2]
        # dropout stuff
        B_U = states_constants[2]
        B_W = states_constants[3]

        x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
        x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
        x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
        x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

        # input gate
        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        # forget gate
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        # memory cell
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        # output
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        # proj_ctx = K.dot(h_tm1, self.Wc_att) + self.b_att
        x_h = K.concatenate((x, h_tm1))
        exp = K.dot(x_h, self.Aw) + self.Ab
        alpha = K.softmax(exp)

        # alpha = K.dot(proj_ctx, self.U_att) + self.c_att
        #
        # h_sampling_mask = K.binomial((1,), p=self.semi_sampling_p, n=1, dtype=x.dtype)
        # alpha = K.softmax(self.temperature * )

        h = o * self.activation(c)
        return h, [h, c]

    def call(self, x, mask=None):
        print('hi')
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape

        initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
