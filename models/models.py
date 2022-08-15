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

"""Defines word2vec model using tf.keras API.
"""

class Word2VecModel(tf.keras.Model):
    """Word2Vec model."""
    def __init__(self,
            unigram_counts, 
            arch='skip_gram',
            algm='negative_sampling', 
            hidden_size=300, 
            batch_size=256, 
            window_size=None,
            max_depth=None,
            negatives=5, 
            power=0.75,
            alpha=0.025,
            min_alpha=0.0001,
            add_bias=True,
            random_seed=0):
        """Constructor.

        Args:
            unigram_counts: a list of ints, the counts of word tokens in the corpus. 
            arch: string scalar, architecture ('skip_gram' or 'cbow').
            algm: string scalar, training algorithm ('negative_sampling' or
                'hierarchical_softmax').
            hidden_size: int scalar, length of word vector.
            batch_size: int scalar, batch size.
            negatives: int scalar, num of negative words to sample.
            power: float scalar, distortion for negative sampling. 
            alpha: float scalar, initial learning rate.
            min_alpha: float scalar, final learning rate.
            add_bias: bool scalar, whether to add bias term to dotproduct 
                between syn0 and syn1 vectors.
            random_seed: int scalar, random_seed.
        """
        super(Word2VecModel, self).__init__()
        self._unigram_counts = unigram_counts
        self._arch = arch
        self._algm = algm
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._window_size = window_size
        self._max_depth = max_depth
        self._negatives = negatives
        self._power = power
        self._alpha = alpha
        self._min_alpha = min_alpha
        self._add_bias = add_bias
        self._random_seed = random_seed

        self._vocab_size = len(unigram_counts)
        self._input_size = (self._vocab_size if self._algm == 'negative_sampling'
                else self._vocab_size - 1)
        self._train_step_signature = self.get_train_step_signature()

        self.add_weight('syn0',
                shape=[self._vocab_size, self._hidden_size],
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.5/self._hidden_size,
                    maxval=0.5/self._hidden_size))
        
        self.add_weight('syn1',
                shape=[self._input_size, self._hidden_size],
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.1, maxval=0.1))

        self.add_weight('biases', 
                shape=[self._input_size],
                initializer=tf.keras.initializers.Zeros()) 

    #@tf.function(input_signature=self._train_step_signature)
    def train_step(self, data):
        inputs, labels, progress = data
        with tf.GradientTape() as tape:
            loss = self((inputs, labels), training=True)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        learning_rate = tf.maximum(self._alpha * (1 - progress[0]) +
                self._min_alpha * progress[0], self._min_alpha)

        if hasattr(gradients[0], '_values'):
            gradients[0]._values *= learning_rate
        else:
            gradients[0] *= learning_rate

        if hasattr(gradients[1], '_values'):
            gradients[1]._values *= learning_rate
        else:
            gradients[1] *= learning_rate

        if hasattr(gradients[2], '_values'):
            gradients[2]._values *= learning_rate
        else:
            gradients[2] *= learning_rate

        self.optimizer.apply_gradients(
                zip(gradients, trainable_vars))

        return {"loss": loss, "learning_rate": learning_rate}

    def call(self, x, training=True):
        """Runs the forward pass to compute loss.

        Args:
            inputs: int tensor of shape [batch_size] (skip_gram) or 
                [batch_size, 2*window_size+1] (cbow) 
            labels: int tensor of shape [batch_size] (negative_sampling) or
                [batch_size, 2*max_depth+1] (hierarchical_softmax)

        Returns:
            loss: float tensor, cross entropy loss. 
        """
        inputs, labels = x
        if self._algm == 'negative_sampling':
            loss = self._negative_sampling_loss(inputs, labels)
        elif self._algm == 'hierarchical_softmax':
            loss = self._hierarchical_softmax_loss(inputs, labels)
        return loss
 
    def _negative_sampling_loss(self, inputs, labels):
        """Builds the loss for negative sampling.

        Args:
            inputs: int tensor of shape [batch_size] (skip_gram) or 
                [batch_size, 2*window_size+1] (cbow)
            labels: int tensor of shape [batch_size]

        Returns:
            loss: float tensor of shape [batch_size, negatives + 1].
        """
        _, syn1, biases = self.weights

        sampled_values = tf.random.fixed_unigram_candidate_sampler(
                true_classes=tf.expand_dims(labels, 1),
                num_true=1,
                num_sampled=self._batch_size*self._negatives,
                unique=True,
                range_max=len(self._unigram_counts),
                distortion=self._power,
                unigrams=self._unigram_counts)

        sampled = sampled_values.sampled_candidates
        sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
        inputs_syn0 = self._get_inputs_syn0(inputs) # [batch_size, hidden_size]
        true_syn1 = tf.gather(syn1, labels) # [batch_size, hidden_size]
        # [batch_size, negatives, hidden_size]
        sampled_syn1 = tf.gather(syn1, sampled_mat)
        # [batch_size]
        true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)
        # [batch_size, negatives]
        sampled_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1), 
                tf.transpose(sampled_syn1, (0, 2, 1)))

        if self._add_bias:
            # [batch_size]
            true_logits += tf.gather(biases, labels)
            # [batch_size, negatives]
            sampled_logits += tf.gather(biases, sampled_mat)

        # [batch_size]
        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(true_logits), logits=true_logits)
        # [batch_size, negatives]
        sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        loss = tf.concat(
                [tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)
        return loss

    def _hierarchical_softmax_loss(self, inputs, labels):
        """Builds the loss for hierarchical softmax.

        Args:
            inputs: int tensor of shape [batch_size] (skip_gram) or 
                [batch_size, 2*window_size+1] (cbow)
            labels: int tensor of shape [batch_size, 2*max_depth+1]

        Returns:
            loss: float tensor of shape [sum_of_code_len]
        """
        _, syn1, biases = self.weights

        inputs_syn0_list = tf.unstack(self._get_inputs_syn0(inputs))
        codes_points_list = tf.unstack(labels)
        max_depth = (labels.shape.as_list()[1] - 1) // 2
        loss = []
        for i in range(self._batch_size):
            inputs_syn0 = inputs_syn0_list[i] # [hidden_size]
            codes_points = codes_points_list[i] # [2*max_depth+1]
            true_size = codes_points[-1]

            codes = codes_points[:true_size]
            points = codes_points[max_depth:max_depth+true_size]
            logits = tf.reduce_sum(
                    tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
            if self._add_bias:
                logits += tf.gather(biases, points)

            # [true_size]
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(codes, 'float32'), logits=logits))
        loss = tf.concat(loss, axis=0)
        return loss

    def _get_inputs_syn0(self, inputs):
        """Builds the activations of hidden layer given input words embeddings 
        `syn0` and input word indices.

        Args:
            inputs: int tensor of shape [batch_size] (skip_gram) or 
                [batch_size, 2*window_size+1] (cbow)

        Returns:
            inputs_syn0: [batch_size, hidden_size]
        """
        # syn0: [vocab_size, hidden_size]
        syn0, _, _ = self.weights
        if self._arch == 'skip_gram':
            inputs_syn0 = tf.gather(syn0, inputs) # [batch_size, hidden_size]
        else:
            inputs_syn0 = []
            contexts_list = tf.unstack(inputs)
            for i in range(self._batch_size):
                contexts = contexts_list[i]
                context_words = contexts[:-1]
                true_size = contexts[-1]
                inputs_syn0.append(
                        tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
            inputs_syn0 = tf.stack(inputs_syn0)

        return inputs_syn0

    def get_config(self):
        return {"unigram_counts": self._unigram_counts,
                "arch": self._arch,
                "algm": self._algm,
                "hidden_size": self._hidden_size,
                "batch_size": self._batch_size,
                "window_size": self._window_size,
                "max_depth": self._max_depth,
                "negatives": self._negatives,
                "power": self._power,
                "alpha": self._alpha,
                "min_alpha": self._min_alpha,
                "add_bias": self._add_bias,
                "random_seed": self._random_seed
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_train_step_signature(self):
        """Get the training step signatures for `inputs`, `labels` and `progress` 
        tensor.
    
        Args:
            arch: string scalar, architecture ('skip_gram' or 'cbow').
            algm: string scalar, training algorithm ('negative_sampling' or
                'hierarchical_softmax').
    
        Returns:
            train_step_signature: a list of three tf.TensorSpec instances,
                specifying the tensor spec (shape and dtype) for `inputs`, `labels` and
                `progress`.
        """
        if self._arch=='skip_gram': 
            inputs_spec = tf.TensorSpec(shape=(self._batch_size,), dtype='int64') 
        elif self._arch == 'cbow':
            inputs_spec = tf.TensorSpec(
                    shape=(self._batch_size, 2*self._window_size+1), dtype='int64')
        else:
            raise ValueError('`arch` must be either "skip_gram" or "cbow".')
    
        if self._algm == 'negative_sampling':
            labels_spec = tf.TensorSpec(shape=(self._batch_size,), dtype='int64') 
        elif self._algm == 'hierarchical_softmax':
            labels_spec = tf.TensorSpec(
                    shape=(self._batch_size, 2*self._max_depth+1), dtype='int64')
        else:
            raise ValueError('`algm` must be either "negative_sampling" or '
                    '"hierarchical_softmax".')
    
        progress_spec = tf.TensorSpec(shape=(self._batch_size,), dtype='float32')
    
        train_step_signature = [inputs_spec, labels_spec, progress_spec]
        return train_step_signature
