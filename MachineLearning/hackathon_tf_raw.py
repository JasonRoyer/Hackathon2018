from __future__ import print_function
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys
from data_handler import *
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


class LSTMUSIC(object):

    def __init__(self, session, scope_name, vocab_size=1,
        num_rnn_layers=2, num_rnn_units=256):

        self.session = session
        self.num_rnn_units = num_rnn_units
        self.num_rnn_layers = num_rnn_layers
        self.vocab_size = vocab_size
        self.scope_name = scope_name

        dt = tf.float32

        with tf.variable_scope(self.scope_name):

            self.feed_in = tf.placeholder(dtype=dt,
                shape=(None, None, vocab_size))
            self.feed_out = tf.placeholder(dtype=dt,
                shape=(None, None, vocab_size))
            self.feed_learning_rate = tf.placeholder(dtype=dt,
                shape=())

            batch_size = tf.shape(self.feed_in)[0]
            seq_length = tf.shape(self.feed_in)[1]
            lr = self.feed_learning_rate

            in_weights = tf.Variable(tf.random_normal((vocab_size, 50)))
            in_biases = tf.Variable(tf.random_normal((50,)))
            #embedding_weights = tf.Variable(char_embedding['arr_0'])
            #embedding_biases = tf.Variable(char_embedding['arr_1'])

            out_weights = tf.Variable(tf.random_normal(
                (num_rnn_units, vocab_size), stddev=0.01))
            out_biases = tf.Variable(tf.random_normal(
                (vocab_size,), stddev=0.01))

            rnn_cell = [tf.contrib.rnn.BasicLSTMCell(num_units=num_rnn_units) \
                for _ in range(num_rnn_layers)]

            rnn_state_size = [cell.state_size for cell in rnn_cell]

            #self.hidden_states = [tf.placeholder(dtype=dt,
            #   shape=(None, s)) for s in rnn_state_size]
            self.hidden_states = []
            for i in range(num_rnn_layers):
                temp_placeholder_1 = tf.placeholder(dtype=tf.float32,
                    shape=(None, num_rnn_units))
                temp_placeholder_2 = tf.placeholder(dtype=tf.float32,
                    shape=(None, num_rnn_units))
                self.hidden_states.append([temp_placeholder_1,
                                           temp_placeholder_2])

            self.rnn_tuple_states = []
            for i in range(num_rnn_layers):
                self.rnn_tuple_states.append(tf.contrib.rnn.LSTMStateTuple(self.hidden_states[i][0], self.hidden_states[i][1]))
            self.rnn_tuple_states = tuple(self.rnn_tuple_states)

            self.multi_rnn = tf.contrib.rnn.MultiRNNCell(rnn_cell)

            num_tiles = batch_size
            expanded_weights = tf.expand_dims(in_weights, axis=0)
            tiled_weights = tf.tile(expanded_weights, [num_tiles, 1, 1])

            in_layer = tf.tanh(
                tf.matmul(self.feed_in, tiled_weights) + in_biases)

            rnn_out_raw, self.rnn_last_state = tf.nn.dynamic_rnn(
                cell=self.multi_rnn, inputs=in_layer,
                initial_state=tuple(self.rnn_tuple_states), dtype=dt)

            rnn_out = tf.tanh(rnn_out_raw)
            rnn_out_flat = tf.reshape(rnn_out, [-1, num_rnn_units])

            linear_flat = tf.matmul(rnn_out_flat, out_weights) + out_biases
            self.linear_out = tf.reshape(linear_flat,
                [batch_size, seq_length, vocab_size])

            feed_out_flat = tf.reshape(self.feed_out, [-1, vocab_size])
            #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #   logits=logits, labels=feed_out_flat))
            self.loss = tf.losses.mean_squared_error(feed_out_flat,
                linear_flat)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

            grads_and_vars = optimizer.compute_gradients(self.loss)
            capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) \
                for grad, var in grads_and_vars]

            self.train_op = optimizer.apply_gradients(capped_grads)


    def train(self, batch_in, batch_out, lr=0.0001):

        batch_size = batch_in.shape[0]
        seq_length = batch_in.shape[1]
        num_rnn_layers = self.num_rnn_layers
        dt = tf.float32

        zero_states = self.session.run(self.multi_rnn.zero_state(batch_size, dt))
        print('batch in shape:', batch_in.shape)
        print('batch out shape:', batch_out.shape)
        feeds = {
            self.feed_in:batch_in,
            self.feed_out:batch_out,
            self.feed_learning_rate:lr,
        }

        for i in range(num_rnn_layers):
            feeds[self.hidden_states[i][0]] = zero_states[i][0]
            feeds[self.hidden_states[i][1]] = zero_states[i][1]

        fetches = [
            self.loss,
            self.train_op
        ]

        loss, _ = self.session.run(fetches, feed_dict=feeds)

        return loss


    def validate(self, valid_in, valid_out):

        batch_size = valid_in.shape[0]
        seq_length = valid_in.shape[1]
        num_rnn_layers = self.num_rnn_layers
        dt = tf.float32
        #num_rnn_units = self.num_rnn_units
        #zero_states = np.zeros((num_rnn_layers, 1, num_hidden_units))
        zero_states = self.session.run(self.multi_rnn.zero_state(batch_size, dt))

        feeds = {
            self.feed_in:valid_in,
            self.feed_out:valid_out,
        }
        #for i in range(num_rnn_layers):
        #    feeds[self.hidden_states[i]] = zero_states[i]
        for i in range(num_rnn_layers):
            feeds[self.hidden_states[i][0]] = zero_states[i][0]
            feeds[self.hidden_states[i][1]] = zero_states[i][1]

        fetches = self.loss

        loss = self.session.run(fetches, feed_dict=feeds)
        #perplexity = np.exp(loss)
        return loss


    def run(self, test_in, num_steps=25):

        batch_size = 1
        seq_length = test_in.shape[0]
        num_rnn_layers = self.num_rnn_layers
        dt = tf.float32

        zero_states = self.session.run(self.multi_rnn.zero_state(batch_size, dt))
        rnn_next_state = zero_states

        probs = []
        feeds = {self.feed_in:test_in}
        fetches = [self.linear_out, self.rnn_last_state]

        for j in range(num_rnn_layers):
            feeds[self.hidden_states[j][0]] = rnn_next_state[j][0]
            feeds[self.hidden_states[j][1]] = rnn_next_state[j][1]
        #print('test in shape', test_in.shape)
        prob, rnn_next_state = self.session.run(fetches, feed_dict=feeds)
        
        for i in range(num_steps):
            #one_hot = soft_prob_to_one_hot(prob[0][-1])
            #probs.append(one_hot)
            probs.append(prob[0][-1])
            
            feeds[self.feed_in] = [[prob[0][-1]]]

            for j in range(num_rnn_layers):
                feeds[self.hidden_states[j][0]] = rnn_next_state[j][0]
                feeds[self.hidden_states[j][1]] = rnn_next_state[j][1]

            prob, rnn_next_state = self.session.run(fetches, feed_dict=feeds)

        print('probs shape:', np.concatenate(probs, axis=0).shape)
        return np.concatenate(probs, axis=0)


    def save(self, filename, save_dir=''):
        save_path = os.path.join(save_dir, filename)
        self.saver.save(self.session, save_path + '.ckpt')


    def set_saver(self, saver):
        self.saver = saver


def decrease_lr(loss, threshold, factor, lr):
    if len(loss) <= 1:
        rate = lr

    else:
        dp = (loss[-2] - loss[-1])/loss[-2]
        if dp < threshold:
            rate = lr * factor
        else:
            rate = lr
    return rate


def train_music(halving_threshold, sequence_length, sample_step,
    max_halvings, max_steps, batch_size, model):
    
    T = sequence_length
    S = sample_step
    M = max_steps
    H = max_halvings
    B = batch_size

    step = 1
    num_halvings = 0
    learning_rate = 0.1
    fs = 44100

    dl = data_loader("music")

    errors = []
    while num_halvings < H and step < M:
        train_in, train_out = dl.get_random_train_batch(B, T)
        loss = model.train(train_in, train_out, learning_rate)

        if ((step % S) == 0) and (step != 0):
            test_in, test_out = dl.get_random_test_batch(B, T)
            #errors.append(model.validate(test_in, test_out))
            #print(errors[-1])

            new_lr = decrease_lr(errors, halving_threshold, 0.5, learning_rate)

            if new_lr != learning_rate:
                num_halvings += 1
                learning_rate = new_lr
                print('Number of halvings:', num_halvings, learning_rate)

            seed = np.zeros([1, 1, 1])
            seed[0, 0, 0] = 1
            generated_audio = model.run(seed, num_steps=80000)

            print('generated audio shape:', generated_audio.shape)
            if len(generated_audio.shape) == 1:
                generated_audio = np.expand_dims(generated_audio, axis=1)
            generated_audio = np.int16((np.abs(generated_audio)/np.max(np.abs(generated_audio)))*32767)
            plt.figure()
            plt.plot(generated_audio)
            #plt.show()
            plt.savefig('wave' + str(step) + '.png')
            generated_audio_upscaled = \
                data_loader.resample(8000, generated_audio, fs)

            wavfile.write('generated' + str(step) + '.wav',
                          fs, generated_audio_upscaled)
        step += 1


def main(argv):
    batch_size = 32
    halving_threshold = 0.003
    sequence_length = 8000
    sample_step = 10
    max_halvings = 15
    max_steps = 20000

    date = datetime.now()
    date = str(date).replace(' ', '').replace(':', '-')
    
    #data_dir = make_dir('data')
    #save_dir = make_dir(date)

    session = tf.Session()
    aimusic = LSTMUSIC(session, 'basiclstm')
    aimusic.set_saver(tf.train.Saver())
    session.run(tf.global_variables_initializer())

    train_music(halving_threshold, sequence_length, sample_step,
        max_halvings, max_steps, batch_size, aimusic)

    #cross_train_lm(halving_threshold, sequence_length, sample_step, 
    #   max_halvings, max_steps, batch_size, corpora, lm, save_dir)

if __name__ == '__main__':
    main(sys.argv[1:])