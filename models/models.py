class deepLAST():
    def __init__(self, options: Options):
        super(deepLAST, self).__init__()
        self.options = options
        
    @classmethod
    def get_model(cls, options: Options) -> tf.keras.Model:
        return cls(options)._node()
   
    def _get_optimizer(self) -> tf.optimizers.Optimizer:
        if self.options.optimizer == "RMSprop":
            optimizer = tf.optimizers.RMSprop(learning_rate=self.options.learning_rate,
                                              momentum=self.options.momentum,
                                              rho=self.options.rho,
                                              epsilon=self.options.epsilon)
        elif self.options.optimizer == "Adam":
            optimizer = tf.optimizers.Adam(learning_rate=self.options.learning_rate,
                                           beta_1=self.options.momentum,
                                           beta_2=self.options.rho,
                                           epsilon=self.options.epsilon)
        else:
            optimizer = self.options.optimizer
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
        
    def _node(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input((2, 93, 100), dtype=tf.float32, name="INP")
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
            attention_result = tf.keras.layers.RepeatVector(93)(attention_result)
            avg = tf.keras.layers.Concatenate()([attention_result, avg])
        else:
            fwd = runit(inputs_fwd)
            rev = runit(inputs_rev)
            avg = tf.keras.layers.Average(name="AV0")([fwd, rev])
        
        logits = tf.keras.layers.Dense(1,
                                       name="FF0",
                                       activation=None)(avg)
        
        flatt = tf.keras.layers.Flatten(name="FL0")(logits)

        output = tf.keras.layers.Dense(1,
                                       name="FF1",
                                       activation='relu')(flatt)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=self._get_optimizer(),
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        return model
