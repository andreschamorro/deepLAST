from typing import Tuple
import tensorflow as tf
from configs.config import Options

class deepLAST():
    def __init__(self, options: Options):
        super(deepLAST, self).__init__()
        self.options = options
        
    @classmethod
    def get_model(cls, options: Options) -> tf.keras.Model:
        return cls(options)._node()

    @staticmethod
    def get_optimizer(options: Options) -> tf.optimizers.Optimizer:
        if options.optimizer == "RMSprop":
            optimizer = tf.optimizers.RMSprop(learning_rate=options.learning_rate,
                                              momentum=options.momentum,
                                              rho=options.rho,
                                              epsilon=options.epsilon)
        elif options.optimizer == "Adam":
            optimizer = tf.optimizers.Adam(learning_rate=options.learning_rate,
                                           beta_1=options.momentum,
                                           beta_2=options.rho,
                                           epsilon=options.epsilon)
        else:
            optimizer = options.optimizer
        return optimizer
    
    def _get_runit_layer(self) -> tf.keras.layers.Layer:
        if self.options.rnn == 'LSTM':
            runit = tf.keras.layers.LSTM(units=self.options.units,
                                         dropout=self.options.dropout,
                                         name='BLSTM',
                                         return_sequences=True)
        else:
            runit = tf.keras.layers.GRU(units=self.options.units,
                                        dropout=self.options.dropout,
                                        name='BGRU',
                                        return_state=self.options.attention,
                                        return_sequences=True)
        return runit

    def _get_input_shape(self) -> Tuple:
        return (2, self.options.seq_size - self.options.k_size + 1, self.options.embedding_dim)
        
    def _node(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(self._get_input_shape(), dtype=tf.float32, name="INP")
        inputs_fwd, inputs_rev = tf.unstack(inputs, axis=1, name="UST")
        runit = self._get_runit_layer()
        if self.options.attention and isinstance(runit, tf.keras.layers.GRU):
            fwd, hidden_fwd = runit(inputs_fwd)
            rev, hidden_rev = runit(inputs_rev)
            hidden = tf.keras.layers.Average(name="AVH")([hidden_fwd, hidden_rev])
            avg = tf.keras.layers.Average(name="AVG")([fwd, rev])
            hidden = tf.keras.layers.Reshape((1, self.options.units),
                                             input_shape=(self.options.units, ))(hidden)
            attention_result = tf.keras.layers.AdditiveAttention()([hidden, avg])
            attention_result = tf.keras.layers.Flatten()(attention_result)
            attention_result = tf.keras.layers.RepeatVector(self._get_input_shape()[1])(attention_result)
            avg = tf.keras.layers.Concatenate()([attention_result, avg])
        else:
            fwd = runit(inputs_fwd)
            rev = runit(inputs_rev)
            avg = tf.keras.layers.Average(name="AV0")([fwd, rev])
        
        logits = tf.keras.layers.Dense(1,
                                       name="FF0",
                                       activation='sigmoid')(avg)
        #soft = tf.keras.layers.Softmax(axis=2)(logits)
        
        flatt = tf.keras.layers.Flatten(name="FL0")(logits)

        output = tf.keras.layers.Dense(1,
                                       name="FF1",
                                       activation='sigmoid')(flatt)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        
        return model
