# coding: utf-8
import numpy as np
import tensorflow as tf

from datetime import datetime

import file_handler as fh
import arguments

#########################################################################################################
################################### LSTM ################################################################
class LSTM(object):
    def __init__(self, args=None):
        if args is None: self.args = arguments.parse_args()
        self.args = args
        # self.saver = tf.train.Saver()
    def train(self, data, target):
        data_size = data.shape[0]
        train_count = int(np.ceil(data_size / self.args.batch_size))

        sh = np.arange(data.shape[0])
        np.random.shuffle(sh)

        train_data = data[sh]
        target = target[sh]

        test_data = train_data[:1000]
        test_target = target[:1000]

        check_time = int(self.args.num_epochs/10)
        con_check_time = check_time-1
        start = datetime.now()
        for i in range(self.args.num_epochs):
            losses = []
            for t in range(train_count):
                s = self.args.batch_size * t
                e = s + self.args.batch_size

                train_data_ = train_data[s:e]
                train_target_ = target[s:e]

                _, loss = self.sess.run((self.train_step, self.loss), feed_dict={self.input: train_data_
                                                                    , self.target: train_target_
                                                                    , self.KeepProbCell: self.keep_prob_cell
                                                                    , self.KeepProbLayer: self.keep_prob_layer})
                losses.append(np.mean(np.nan_to_num(loss)))
            if i%check_time == con_check_time:
                predicts, rmse = self.sess.run((self.prediction, self.rmse), feed_dict={self.input: test_data, self.target: test_target, self.KeepProbCell: 1, self.KeepProbLayer: 1})
                # accuracy, precision, recall, f1 = eh.evaluatePredictions(test_target, predicts)
                print('=====================================================================================================================================================')
                # print('epoch %d: loss %03.5f rmse: %03.5f accuracy : %.4f, precision : %.4f, recall : %.4f, f1-measure : %.4f' % (i+1, np.mean(losses), rmse, accuracy, precision, recall, f1))
                print('epoch %d: loss %03.9f rmse: %03.5f' % (i + 1, np.mean(losses), rmse))
                print(datetime.now()-start)
                print('=====================================================================================================================================================')
                fh.saveTxT(predicts.reshape(predicts.shape[0], 1), 'predicts/epoch_%d' % (i + 1))
                start = datetime.now()
        return np.mean(losses)

    def predict(self, data):
        predicts = self.sess.run((self.prediction), feed_dict={self.input: data, self.KeepProbCell: self.args.keep_prob_cell, self.KeepProbLayer: self.args.keep_prob_layer})
        return predicts
    def evaluation(self, data, target):
        rmse = self.sess.run((self.rmse), feed_dict={self.input: data, self.target: target, self.KeepProbCell: 1,self.KeepProbLayer: 1})
        return rmse

    def init_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = tf.ConfigProto(gpu_options=gpu_options)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess)

    def generateModel(self, name, input, output_size, activation, n_layers, hidden_size, reuse):
        def rnn_cell(): return tf.nn.rnn_cell.BasicLSTMCell(hidden_size, activation=activation, reuse=reuse)
        # with tf.variable_scope('LSTM_'+name, reuse=reuse):
        Cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(rnn_cell(), input_keep_prob=self.KeepProbCell) for _ in range(n_layers)]
            , state_is_tuple=True)
        Cell = tf.contrib.rnn.DropoutWrapper(Cell, input_keep_prob=self.KeepProbLayer)

        # Create RNN
        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of RNNCells containing a
        reverse_input = tf.reverse(input, axis=[1], name='reverse_input')
        Output, State = tf.nn.dynamic_rnn(Cell, reverse_input, dtype=tf.float32)
        h = tf.reverse(Output, axis=[1])

        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)
        w = tf.get_variable("w", [h.get_shape()[2], output_size], initializer=w_init)
        b = tf.get_variable("b", [output_size], initializer=b_init)

        h = tf.transpose(h, (1, 0, 2))

        e = []
        for i in range(h.get_shape()[0]) :
            e.append(tf.matmul(h[i], w) + b)
        return tf.transpose(e, (1, 0, 2), name=name)

    def generateModels(self, input_size, output_size, step_size):
        tf.reset_default_graph()

        self.input = tf.placeholder(tf.float32, [None, step_size, input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, step_size, output_size], name='target')

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = tf.placeholder(tf.float32, [], name='KeepProbCell')
        self.KeepProbLayer = tf.placeholder(tf.float32, [], name='KeepProbLayer')

        # generating alpha values
        self.alpha = self.generateModel(name='layers', input=self.input, output_size=1
                                                , activation=tf.nn.elu, n_layers=self.args.n_layers
                                                , hidden_size=self.args.n_hidden, reuse=False)

        self.prediction = self.alpha
        target_shape = tf.shape(self.target)
        predict_shape = tf.shape(self.prediction)
        target = tf.reshape(self.target, [target_shape[0], target_shape[1]*target_shape[2]])
        predict = tf.reshape(self.prediction, [predict_shape[0], predict_shape[1]*predict_shape[2]])
        target = tf.reduce_sum(target, axis=1)
        predict = tf.reduce_sum(predict, axis=1)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate, name='optimizer')
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate, name='optimizer')
        self.loss = tf.losses.mean_squared_error(target, predict)
        # self.loss = -tf.reduce_mean(target*tf.log(predict)+(1-target)*tf.log(1-predict))
        # self.loss = tf.reduce_mean(tf.square(target - predict))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(target - predict)), name='rmse')
        self.train_step = self.optimizer.minimize(self.loss, name='train_step')

        self.init_session()

    def save(self, model_path):
        self.saver.save(self.sess, model_path, global_step=1000)
    def restore(self, model_path):
        self.init_session()
        self.saver = tf.train.import_meta_graph(model_path+'-1000.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path[:model_path.rfind('/')+1]))

        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.target = graph.get_tensor_by_name("target:0")

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = graph.get_tensor_by_name('KeepProbCell:0')
        self.KeepProbLayer = graph.get_tensor_by_name('KeepProbLayer:0')

        self.prediction = graph.get_tensor_by_name('predict:0')
        layer_name = 'layers'
        self.alpha = graph.get_tensor_by_name('LSTM_%s:%s:0'%(layer_name, layer_name))
        # self.emb = graph.get_tensor_by_name('wv:0')
        # self.wy = graph.get_tensor_by_name('wy:0')

        self.optimizer = graph.get_tensor_by_name("optimizer:0")
        self.loss = graph.get_tensor_by_name("loss:0")
        self.rmse = graph.get_tensor_by_name("rmse:0")
        self.train_step = graph.get_tensor_by_name("train_step:0")

class Bidirectional_LSTM(object):
    def __init__(self, args):
        self.args = args
        # self.saver = tf.train.Saver()
    def train(self, data, target):
        data_size = data.shape[0]
        train_count = int(np.ceil(data_size / self.args.batch_size))

        sh = np.arange(data.shape[0])
        np.random.shuffle(sh)

        train_data = data[sh]
        target = target[sh]

        test_data = train_data[:1000]
        test_target = target[:1000]

        check_time = int(self.args.num_epochs/10)
        con_check_time = check_time-1
        start = datetime.now()
        for i in range(self.args.num_epochs):
            losses = []
            for t in range(train_count):
                s = self.args.batch_size * t
                e = s + self.args.batch_size

                train_data_ = train_data[s:e]
                train_target_ = target[s:e]

                _, loss = self.sess.run((self.train_step, self.loss), feed_dict={self.input: train_data_
                                                                    , self.target: train_target_
                                                                    , self.KeepProbCell: self.keep_prob_cell
                                                                    , self.KeepProbLayer: self.keep_prob_layer})
                losses.append(np.mean(np.nan_to_num(loss)))
            if i%check_time == con_check_time:
                predicts, rmse = self.sess.run((self.prediction, self.rmse), feed_dict={self.input: test_data, self.target: test_target, self.KeepProbCell: 1, self.KeepProbLayer: 1})
                # accuracy, precision, recall, f1 = eh.evaluatePredictions(test_target, predicts)
                print('=====================================================================================================================================================')
                # print('epoch %d: loss %03.5f rmse: %03.5f accuracy : %.4f, precision : %.4f, recall : %.4f, f1-measure : %.4f' % (i+1, np.mean(losses), rmse, accuracy, precision, recall, f1))
                print('epoch %d: loss %03.9f rmse: %03.5f' % (i + 1, np.mean(losses), rmse))
                print(datetime.now()-start)
                print('=====================================================================================================================================================')
                fh.saveTxT(predicts.reshape(predicts.shape[0], 1), 'predicts/epoch_%d' % (i + 1))
                start = datetime.now()
        return np.mean(losses)

    def predict(self, data):
        predicts = self.sess.run((self.prediction), feed_dict={self.input: data, self.KeepProbCell: self.args.keep_prob_cell, self.KeepProbLayer: self.args.keep_prob_layer})
        return predicts
    def evaluation(self, data, target):
        rmse = self.sess.run((self.rmse), feed_dict={self.input: data, self.target: target, self.KeepProbCell: 1,self.KeepProbLayer: 1})
        return rmse

    def init_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = tf.ConfigProto(gpu_options=gpu_options)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess)

    def generateModel(self, name, input, output_size, activation, n_layers, hidden_size, reuse):
        def rnn_cell(): return tf.nn.rnn_cell.LSTMCell(num_units = hidden_size, activation=activation, state_is_tuple = True)
        # with tf.variable_scope('LSTM_'+name, reuse=reuse):
        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(rnn_cell(), input_keep_prob=self.KeepProbCell) for _ in range(n_layers)]
            , state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.KeepProbLayer)

        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(rnn_cell(), input_keep_prob=self.KeepProbCell) for _ in range(n_layers)]
            , state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.KeepProbLayer)

        # Create RNN
        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of RNNCells containing a
        outputs, State = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)

        outputs_fw = tf.transpose(outputs[0], [1, 0, 2])
        outputs_bw = tf.transpose(outputs[1], [1, 0, 2])

        W = tf.Variable(tf.random_normal([hidden_size * 2, output_size]))
        b = tf.Variable(tf.random_normal([output_size]))

        outputs_concat = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1)

        pred = tf.matmul(outputs_concat, W) + b

        return pred

    def generateModels(self, input_size, output_size, step_size):
        tf.reset_default_graph()

        self.input = tf.placeholder(tf.float32, [None, step_size, input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, output_size], name='target')

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = tf.placeholder(tf.float32, [], name='KeepProbCell')
        self.KeepProbLayer = tf.placeholder(tf.float32, [], name='KeepProbLayer')

        # generating alpha values
        self.prediction = self.generateModel(name='layers', input=self.input, output_size=output_size
                                                , activation=tf.nn.elu, n_layers=self.args.n_layers
                                                , hidden_size=self.args.n_hidden, reuse=False)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate, name='optimizer')
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate, name='optimizer')
        self.loss = tf.losses.mean_squared_error(self.target, self.prediction)
        # self.loss = -tf.reduce_mean(target*tf.log(predict)+(1-target)*tf.log(1-predict))
        # self.loss = tf.reduce_mean(tf.square(target - predict))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.target-self.prediction)), name='rmse')
        self.train_step = self.optimizer.minimize(self.loss, name='train_step')

        self.init_session()

    def save(self, model_path):
        self.saver.save(self.sess, model_path, global_step=1000)
    def restore(self, model_path):
        self.init_session()
        self.saver = tf.train.import_meta_graph(model_path+'-1000.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path[:model_path.rfind('/')+1]))

        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.target = graph.get_tensor_by_name("target:0")

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = graph.get_tensor_by_name('KeepProbCell:0')
        self.KeepProbLayer = graph.get_tensor_by_name('KeepProbLayer:0')

        self.prediction = graph.get_tensor_by_name('predict:0')
        layer_name = 'layers'
        self.alpha = graph.get_tensor_by_name('LSTM_%s:%s:0'%(layer_name, layer_name))
        # self.emb = graph.get_tensor_by_name('wv:0')
        # self.wy = graph.get_tensor_by_name('wy:0')

        self.optimizer = graph.get_tensor_by_name("optimizer:0")
        self.loss = graph.get_tensor_by_name("loss:0")
        self.rmse = graph.get_tensor_by_name("rmse:0")
        self.train_step = graph.get_tensor_by_name("train_step:0")

#########################################################################################################
################################### CNN #################################################################
class CNN(object):
    def __init__(self, args):
        self.args = args

    # Create the neural network
    def conv_net(self, x, shape, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, self.image_h, self.image_w, self.kernel])

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)

        return out
    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = self.conv_net(features, features.shape, self.args.output_size, self.args.keep_prob, reuse=False,
                                is_training=True)
        logits_test = self.conv_net(features, features.shape, self.args.output_size, self.args.keep_prob, reuse=True,
                               is_training=False)

        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    def generateModels(self, image_h, image_w, kernel, output_size):
        self.model = tf.estimator.Estimator(self.model_fn)
        self.image_h, self.image_w, self.kernel, self.output_size = image_h, image_w, kernel, output_size

    def evaluation(self, data, target):
        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = self.model.evaluate(input_fn)
        # e={'accuracy': 0.9846, 'loss': 0.052267317, 'global_step': 2000}
        return e
    def train(self, data, target):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, num_epochs=self.args.num_epochs, shuffle=True)
        # Train the Model
        self.model.train(input_fn, steps=self.args.num_epochs)

    def predict(self, data):
        # Define the input function for training
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, num_epochs=1, shuffle=False)
        # Train the Model
        return self.model.predict(predict_input_fn)

#########################################################################################################
################################### ResNet ##############################################################
class ResNet(object):
    def __init__(self, args):
        self.args = args

        self.BATCH_NORM_DECAY = 0.9
        self.BATCH_NORM_EPSILON = 1e-5
    def batch_norm_relu(self, inputs, is_training, relu=True, init_zero=False, data_format='channels_first'):
        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()

        if data_format == 'channels_first':
            axis = 1
        else:
            axis = 3

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            momentum=self.BATCH_NORM_DECAY,
            epsilon=self.BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=is_training,
            fused=True,
            gamma_initializer=gamma_initializer)

        if relu:
            inputs = tf.nn.relu(inputs)
        return inputs

    def fixed_padding(self, inputs, kernel_size, data_format='channels_first'):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])

        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format='channels_first'):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format=data_format)

        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def residual_block(self, inputs, filters, is_training, strides, use_projection=False, data_format='channels_first'):
        shortcut = inputs
        if use_projection:
            # Projection shortcut in first layer to match filters and strides
            shortcut = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=strides,
                data_format=data_format)
            shortcut = self.batch_norm_relu(shortcut, is_training, relu=False,
                                       data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                      strides=strides, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, is_training, data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                      strides=1, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, is_training, data_format=data_format,
                                 relu=False, init_zero=True)

        return tf.nn.relu(inputs + shortcut)

    def bottleneck_block(self, inputs, filters, is_training, strides, use_projection=False, data_format='channels_first'):
        shortcut = inputs
        if use_projection:
            # Projection shortcut only in first block within a group. Bottleneck blocks
            # end with 4 times the number of filters.
            filters_out = 4 * filters
            shortcut = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                data_format=data_format)
            shortcut = self.batch_norm_relu(shortcut, is_training, relu=False,
                                       data_format=data_format)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)
        inputs = self.batch_norm_relu(inputs, is_training, data_format=data_format)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
        inputs = self.batch_norm_relu(inputs, is_training, data_format=data_format)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)
        inputs = self.batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                                 data_format=data_format)

        return tf.nn.relu(inputs + shortcut)

    def block_group(self, inputs, filters, block_fn, blocks, strides, is_training, name, data_format='channels_first'):
        # Only the first block per block_group uses projection shortcut and strides.
        inputs = block_fn(inputs, filters, is_training, strides,
                          use_projection=True, data_format=data_format)

        for _ in range(1, blocks):
            inputs = block_fn(inputs, filters, is_training, 1,
                              data_format=data_format)

        return tf.identity(inputs, name)

    def resnet_v1_generator(self, block_fn, layers, num_classes, data_format='channels_first'):
        def model(inputs, is_training):
            """Creation of the model graph."""
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=64, kernel_size=7, strides=2,
                data_format=data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = self.batch_norm_relu(inputs, is_training, data_format=data_format)

            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=3, strides=2, padding='SAME',
                data_format=data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

            inputs = self.block_group(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
                strides=1, is_training=is_training, name='block_group1',
                data_format=data_format)
            inputs = self.block_group(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
                strides=2, is_training=is_training, name='block_group2',
                data_format=data_format)
            inputs = self.block_group(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
                strides=2, is_training=is_training, name='block_group3',
                data_format=data_format)
            inputs = self.block_group(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
                strides=2, is_training=is_training, name='block_group4',
                data_format=data_format)

            # The activation is 7x7 so this is a global average pool.
            # TODO(huangyp): reduce_mean will be faster.
            pool_size = (inputs.shape[1], inputs.shape[2])
            inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=pool_size, strides=1, padding='VALID',
                data_format=data_format)
            inputs = tf.identity(inputs, 'final_avg_pool')
            inputs = tf.reshape(
                inputs, [-1, 2048 if block_fn is self.bottleneck_block else 512])
            inputs = tf.layers.dense(
                inputs=inputs,
                units=num_classes,
                kernel_initializer=tf.random_normal_initializer(stddev=.01))
            inputs = tf.identity(inputs, 'final_dense')
            return inputs

        model.default_image_size = 224
        return model

    def resnet_v1(self, resnet_depth, num_classes, data_format='channels_first'):
        """Returns the ResNet model for a given size and number of output classes."""
        model_params = {
            18: {'block': self.residual_block, 'layers': [2, 2, 2, 2]},
            34: {'block': self.residual_block, 'layers': [3, 4, 6, 3]},
            50: {'block': self.bottleneck_block, 'layers': [3, 4, 6, 3]},
            101: {'block': self.bottleneck_block, 'layers': [3, 4, 23, 3]},
            152: {'block': self.bottleneck_block, 'layers': [3, 8, 36, 3]},
            200: {'block': self.bottleneck_block, 'layers': [3, 24, 36, 3]}
        }

        if resnet_depth not in model_params:
            raise ValueError('Not a valid resnet_depth:', resnet_depth)

        params = model_params[resnet_depth]
        return self.resnet_v1_generator(
            params['block'], params['layers'], num_classes, data_format)

    # Create the neural network
    def conv_net(self, x_dict, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['images']

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, self.image_h, self.image_w, self.kernel])

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)

        return out
    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = self.conv_net(features, self.args.output_size, self.args.keep_prob, reuse=False,
                                is_training=True)
        logits_test = self.conv_net(features, self.args.output_size, self.args.keep_prob, reuse=True,
                               is_training=False)

        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    def generateModels(self, image_h, image_w, kernel, output_size):
        self.model = tf.estimator.Estimator(self.model_fn)
        self.image_h, self.image_w, self.kernel, self.output_size = image_h, image_w, kernel, output_size

    def evaluation(self, data, target):
        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = self.model.evaluate(input_fn)
        # e={'accuracy': 0.9846, 'loss': 0.052267317, 'global_step': 2000}
        return e
    def train(self, data, target):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, num_epochs=self.args.num_epochs, shuffle=True)
        # Train the Model
        self.model.train(input_fn, steps=self.args.num_epochs)

    def predict(self, data):
        # Define the input function for training
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, num_epochs=1, shuffle=False)
        # Train the Model
        return self.model.predict(predict_input_fn)

#########################################################################################################
################################### AlexNet #############################################################
class AlexNet(object):
    def __init__(self, args):
        self.args = args

    # Create the neural network
    def conv_net(self, x_dict, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['images']

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, self.image_h, self.image_w, self.kernel])

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)

        return out
    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = self.conv_net(features, self.args.output_size, self.args.keep_prob, reuse=False,
                                is_training=True)
        logits_test = self.conv_net(features, self.args.output_size, self.args.keep_prob, reuse=True,
                               is_training=False)

        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    def generateModels(self, image_h, image_w, kernel, output_size):
        self.model = tf.estimator.Estimator(self.model_fn)
        self.image_h, self.image_w, self.kernel, self.output_size = image_h, image_w, kernel, output_size

    def evaluation(self, data, target):
        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = self.model.evaluate(input_fn)
        # e={'accuracy': 0.9846, 'loss': 0.052267317, 'global_step': 2000}
        return e
    def train(self, data, target):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, num_epochs=self.args.num_epochs, shuffle=True)
        # Train the Model
        self.model.train(input_fn, steps=self.args.num_epochs)

    def predict(self, data):
        # Define the input function for training
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, num_epochs=1, shuffle=False)
        # Train the Model
        return self.model.predict(predict_input_fn)

#########################################################################################################
################################### VGG #################################################################
class VGG(object):
    def __init__(self, args):
        self.args = args

    # Create the neural network
    def conv_net(self, x_dict, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['images']

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, self.image_h, self.image_w, self.kernel])

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)

        return out
    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = self.conv_net(features, self.args.output_size, self.args.keep_prob, reuse=False,
                                is_training=True)
        logits_test = self.conv_net(features, self.args.output_size, self.args.keep_prob, reuse=True,
                               is_training=False)

        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    def generateModels(self, image_h, image_w, kernel, output_size):
        self.model = tf.estimator.Estimator(self.model_fn)
        self.image_h, self.image_w, self.kernel, self.output_size = image_h, image_w, kernel, output_size

    def evaluation(self, data, target):
        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = self.model.evaluate(input_fn)
        # e={'accuracy': 0.9846, 'loss': 0.052267317, 'global_step': 2000}
        return e
    def train(self, data, target):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, y=target,
            batch_size=self.args.batch_size, num_epochs=self.args.num_epochs, shuffle=True)
        # Train the Model
        self.model.train(input_fn, steps=self.args.num_epochs)

    def predict(self, data):
        # Define the input function for training
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': data}, num_epochs=1, shuffle=False)
        # Train the Model
        return self.model.predict(predict_input_fn)

def getML(args):
    return {
        'CNN':CNN(args), 'ResNet':ResNet(args),
        'AlexNet':AlexNet(args), 'VGG':VGG(args), 'LSTM':LSTM(args)
    }[args.model_name]