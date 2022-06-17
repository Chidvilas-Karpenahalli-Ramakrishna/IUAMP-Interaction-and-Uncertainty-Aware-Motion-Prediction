# Required built-in modules:
# ----------------------------
import random
import shutil
import sys
import datetime
import json
import os
from pathlib import Path

# Required additional libraries:
# --------------------------------
import numpy as np
from PIL import Image
from tqdm import tqdm

# Required TensorFlow modules:
# ----------------------------------
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

# To prevent TensorFlow log outputs:
# ---------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Train-test split:
# ------------------
def data_divider(number_of_test_samples, validation_data_percentage, data_path,
                 random_seed):
    """
    Function to perform train-test split

    :param number_of_test_samples: Total number of test samples needed

    :param validation_data_percentage: Percentage of data required as validation
    data during training

    :param data_path: Path where BEVs and OGMs are stored

    :param random_seed: Random-seed value to reproduce the train-test split
    """
    x_scenes = None
    y_scenes = None

    try:
        x_path = os.path.join(data_path, "bev_nuscenes")
        x_scenes = os.listdir(x_path)
        y_path = os.path.join(data_path, "ogm_nuscenes")
        y_scenes = os.listdir(y_path)
    except FileNotFoundError:
        print("Please check the path for x-validate and y-validate directories.")
    
    # Shuffle the dataset:
    # -----------------------
    random.Random(random_seed).shuffle(x_scenes)
    random.Random(random_seed).shuffle(y_scenes)
    
    # Raise error if there is a mismatch:
    # ---------------------------------------
    if x_scenes != y_scenes:
        sys.exit("There is a mismatch in the dataset. "
                 "The number of X (inputs) and y (targets) are not same. "
                 "Exiting the data-dividing process.")
    else:
        print("BEV and OGM scenes match. Continuing with the data-divider process.")

    output_path = os.path.join(data_path, "data")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    print("Successfully created data folder for train and validation data.")

    test_scenes = x_scenes[-number_of_test_samples:]
    x_scenes = x_scenes[:-number_of_test_samples]
    train_index = int(len(x_scenes) - np.ceil(len(x_scenes) * (validation_data_percentage / 100)))
    train_scenes = x_scenes[:train_index]
    validation_scenes = x_scenes[train_index:]

    # Training data:
    # ---------------
    os.makedirs(os.path.join(output_path, "x_train"))
    os.makedirs(os.path.join(output_path, "y_train"))
    for each_scene in tqdm(train_scenes, desc="Copying training data",
                           colour="green"):
        source_x_train = os.path.join(data_path, "bev_nuscenes", each_scene)
        source_y_train = os.path.join(data_path, "ogm_nuscenes", each_scene)
        destination_x_train = os.path.join(output_path, "x_train", each_scene)
        destination_y_train = os.path.join(output_path, "y_train", each_scene)
        shutil.copytree(source_x_train, destination_x_train)
        shutil.copytree(source_y_train, destination_y_train)

    # validation data:
    # -----------------
    os.makedirs(os.path.join(output_path, "x_validate"))
    os.makedirs(os.path.join(output_path, "y_validate"))
    for each_scene in tqdm(validation_scenes, desc="Copying validation data",
                           colour="blue"):
        source_x_validate = os.path.join(data_path, "bev_nuscenes", each_scene)
        source_y_validate = os.path.join(data_path, "ogm_nuscenes", each_scene)
        destination_x_train = os.path.join(output_path, "x_validate", each_scene)
        destination_y_train = os.path.join(output_path, "y_validate", each_scene)
        shutil.copytree(source_x_validate, destination_x_train)
        shutil.copytree(source_y_validate, destination_y_train)

    # test data:
    # -----------
    os.makedirs(os.path.join(output_path, "x_test"))
    os.makedirs(os.path.join(output_path, "y_test"))
    for each_scene in tqdm(test_scenes, desc="Copying test data",
                           colour="red"):
        source_x_test = os.path.join(data_path, "bev_nuscenes", each_scene)
        source_y_test = os.path.join(data_path, "ogm_nuscenes", each_scene)
        destination_x_test = os.path.join(output_path, "x_test", each_scene)
        destination_y_test = os.path.join(output_path, "y_test", each_scene)
        shutil.copytree(source_x_test, destination_x_test)
        shutil.copytree(source_y_test, destination_y_test)


# Manual data generator to feed data to our machine learning model:
# --------------------------------------------------------------------------
# noinspection PyTypeChecker
def data_generator(desired_shape, input_data_path, time_steps, skip_steps, batch_size,
                   random_seed, validate=False):
    """
    A manual data-generator for the current task. Iterates through BEVs and
    OGMs and generates a 5D tensor which is passed as input to the model

    :param desired_shape: Desired shape of the input

    :param input_data_path: Path to BEVs and OGMs

    :param time_steps: The length of the BEV and OGM sequences

    :param skip_steps: Number of BEVs and OGMs to skip to generate the next sequences

    :param batch_size: Batch-size of the input

    :param random_seed: A random-seed to reproduce the same inputs

    :param validate: Takes True or False values. If data_generator is being used
    for training keep False. If used for testing, keep True.

    :return: a 5D tensor of shape (batch_size, time_steps, height, width, channels)
    """
    while True:
        x_path = None
        y_path = None
        x_scenes = None
        y_scenes = None
        data_path = os.path.join(input_data_path, 'data')

        if validate:
            try:
                x_path = os.path.join(data_path, 'x_validate')
                x_scenes = os.listdir(x_path)
                y_path = os.path.join(data_path, 'y_validate')
                y_scenes = os.listdir(y_path)
            except FileNotFoundError:
                print("Please check the path for x-validate and y-validate directories.")
        else:
            try:
                x_path = os.path.join(data_path, 'x_train')
                x_scenes = os.listdir(x_path)
                y_path = os.path.join(data_path, 'y_train')
                y_scenes = os.listdir(y_path)
            except FileNotFoundError:
                print("Please check the path for x-train and y-train directories.")

        random.Random(random_seed).shuffle(x_scenes)
        random.Random(random_seed).shuffle(y_scenes)

        if x_scenes != y_scenes:
            print("There is a mismatch in the dataset. "
                  "The number of X (inputs) and y (targets) are not same. Please check the data folder.")
            break

        for each_x_scene, each_y_scene in zip(x_scenes, y_scenes):
            x_samples = sorted(os.listdir(os.path.join(x_path, each_x_scene)),
                               key=lambda item: int(item.split('_')[0]))
            y_samples = sorted(os.listdir(os.path.join(y_path, each_y_scene)),
                               key=lambda item: int(item.split('_')[0]))

            if len(x_samples) != len(y_samples):
                print(f"There is a mismatch in the number of input and target samples in {each_x_scene}. "
                      "Please check the scene. Continuing with other scenes.")
                continue

            x_batch = []
            y_batch = []
            count = 0

            for i in range(0, len(x_samples) - 2 * time_steps + 1):
                if not i % skip_steps:
                    count += 1
                    x = x_samples[i: i + time_steps]
                    y = y_samples[i + time_steps: i + 2 * time_steps]
                    x_single_batch = []
                    y_single_batch = []

                    for each_x_sample, each_y_sample in zip(x, y):
                        '''
                        As it can be seen. The data is also being normalized simultaneously.
                        '''
                        x_single_batch.append((np.asarray(Image.open(
                            os.path.join(x_path, each_x_scene, each_x_sample)).resize(
                            desired_shape, Image.NEAREST))[:, :, :3]) / 255)
                        y_single_batch.append(np.load(os.path.join(y_path, each_y_scene, each_y_sample)))

                    x_batch.append(np.asarray(x_single_batch))
                    y_batch.append(np.asarray(y_single_batch))

                    if count == batch_size:
                        yield np.asarray(x_batch), np.asarray(y_batch)
                        x_batch = []
                        y_batch = []
                        count = 0


# Function to count the number of tests possible per epoch:
# ----------------------------------------------------------------
def steps_per_epoch_counter(input_data_path, time_steps, skip_steps, batch_size,
                            random_seed, validate=False, steps_count=0):
    """
    A function to calculate the steps possible per epoch. Has to be passed to the
    model to inform the model when one epoch in completed

    :param input_data_path: Path to BEVs and OGMs

    :param time_steps: Length of the sequence

    :param skip_steps: Number of BEVs or OGMs to skip between inputs

    :param batch_size: Batch-size of the input

    :param random_seed: Random-seed to replicate the results

    :param validate: Keep False for training and True during testing

    :param steps_count: Initialized to 0. Updated as the code iterates through
    the inputs

    :return: steps_count: Possible steps per epoch for the given data
    """
    while True:
        x_path = None
        y_path = None
        x_scenes = None
        y_scenes = None
        data_path = os.path.join(input_data_path, 'data')

        if validate:
            try:
                x_path = os.path.join(data_path, 'x_validate')
                x_scenes = os.listdir(x_path)
                y_path = os.path.join(data_path, 'y_validate')
                y_scenes = os.listdir(y_path)
            except FileNotFoundError:
                print("Please check the path for x-validate and y-validate directories.")
        else:
            try:
                x_path = os.path.join(data_path, 'x_train')
                x_scenes = os.listdir(x_path)
                y_path = os.path.join(data_path, 'y_train')
                y_scenes = os.listdir(y_path)
            except FileNotFoundError:
                print("Please check the path for x-train and y-train directories.")

        random.Random(random_seed).shuffle(x_scenes)
        random.Random(random_seed).shuffle(y_scenes)

        if x_scenes != y_scenes:
            print("There is a mismatch in the dataset. "
                  "The number of X (inputs) and y (targets) are not same. Please check the data folder.")
            break

        for each_x_scene, each_y_scene in zip(x_scenes, y_scenes):
            x_samples = sorted(os.listdir(os.path.join(x_path, each_x_scene)),
                               key=lambda item: int(item.split('_')[0]))
            y_samples = sorted(os.listdir(os.path.join(y_path, each_y_scene)),
                               key=lambda item: int(item.split('_')[0]))

            if len(x_samples) != len(y_samples):
                print(f"There is a mismatch in the number of input and target samples in {each_x_scene}. "
                      "Please check the x and y directories. Continuing with other scenes.")
                continue

            x_batch = []
            y_batch = []
            count = 0

            for i in range(0, len(x_samples) - 2 * time_steps + 1):
                if not i % skip_steps:
                    count += 1
                    x = x_samples[i: i + time_steps]
                    y = y_samples[i + time_steps: i + 2 * time_steps]
                    x_single_batch = []
                    y_single_batch = []
                    for each_x_sample, each_y_sample in zip(x, y):
                        x_single_batch.append(each_x_sample)
                        y_single_batch.append(each_y_sample)

                    x_batch.append(np.asarray(x_single_batch))
                    y_batch.append(np.asarray(y_single_batch))

                    if count == batch_size:
                        x_batch = []
                        y_batch = []
                        count = 0
                        steps_count += 1

        return steps_count


# Function to initialize the model parameters:
# ------------------------------------------------
def model_initializer(log_path):
    """
    A function to initialize the model with hyperparameters and other model
    parameters

    :param log_path:  Path to store the log-files during training

    :return: Returns the conditions for early-stopping, conditions to reduce
    learning rate, tensorboard callback and condition for model checkpoint
    """
    try:
        # Load the model parameters file defined by the user:
        # ----------------------------------------------------------
        model_parameters_path = os.path.join(os.path.dirname(__file__), 'model_parameters.json')

        with open(Path(model_parameters_path)) as model_parameters_file:
            model_parameters = json.load(model_parameters_file)
            print("Successfully opened model parameters file.")

            # Define log paths:
            # -------------------
            start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(log_path, "train_logs", "logs", "fit" + "_" + start_time)
            checkpoint_dir = os.path.join(log_path, "train_logs", "checkpoints", "model" + "_" + start_time)

            early_stopping = EarlyStopping(monitor=model_parameters["term_to_monitor"],
                                           patience=model_parameters["early_stopping_patience"],
                                           verbose=1, mode=model_parameters["mode_to_monitor"],
                                           min_delta=model_parameters["min_delta"])

            reduce_lr = ReduceLROnPlateau(monitor=model_parameters["term_to_monitor"],
                                          patience=model_parameters["reduce_learning_rate_patience"],
                                          factor=model_parameters["learning_rate_reduction_factor"],
                                          verbose=1, min_lr=model_parameters["minimum_learning_rate"],
                                          min_delta=model_parameters["min_delta"],
                                          mode=model_parameters["mode_to_monitor"])

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                                           monitor=model_parameters["term_to_monitor"],
                                                                           mode=model_parameters["mode_to_monitor"],
                                                                           save_best_only=True, save_weights_only=False)
            print("Successfully parameterized the model. Continuing with training.")

    except FileNotFoundError:
        print("The .json file containing the model parameters was not found. Please check the folder."
              "Model parameterization failed.")

    return early_stopping, reduce_lr, tensorboard_callback, model_checkpoint_callback


# Sequence self-attention mechanism:
# ----------------------------------------
class SeqSelfAttention(keras.layers.Layer):
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.
        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e -= 10000.0 * (1.0 - K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx()))
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= 10000.0 * ((1.0 - mask) * (1.0 - K.permute_dimensions(mask, (0, 2, 1))))

        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


# Novel Combi-loss function (Combination of Binary Crossentropy and Tversky loss):
# -------------------------------------------------------------------------------------------
def cross_entropy_tversky_loss(y_true, y_pred):
    """
    Novel Combi-loss function that combined Tversky loss and Binary Crossentropy

    :param y_true: Target for the model

    :param y_pred: Model predictions

    :return: Combi-loss of y-true and y-prediction
    """

    def tversky_loss_prime(tversky_y_true, tversky_y_pred):
        """
        Tversky loss

        :param tversky_y_true: Target

        :param tversky_y_pred: Model prediction

        :return: Tversky part of the loss
        """

        # Please change the Tversky-beta parameter below as per requirements:
        # -------------------------------------------------------------------------------
        beta = 0.1

        y_pred_prime = tf.math.sigmoid(tversky_y_pred)
        numerator = tversky_y_true * y_pred_prime
        denominator = tversky_y_true * y_pred_prime + beta * (1 - tversky_y_true) * y_pred_prime \
                      + (1 - beta) * tversky_y_true * (1 - y_pred_prime)

        return 1 - (tf.reduce_sum(numerator) / tf.reduce_sum(denominator))

    """
    Combining Tversky loss with Binary Crossentropy to obtain Combi-loss
    """
    y_true_prime = tf.cast(y_true, tf.float32)
    combined_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true_prime, y_pred) + tversky_loss_prime(y_true_prime,
                                                                                                       y_pred)
    return tf.reduce_mean(combined_loss)


# Permanent Dropout on Keras:
# --------------------------------
def permanent_dropout(rate):
    """
    Function to bring about Monte Carlo (MC) Dropout

    :param rate: Percentage of neurons to drop per layer
    
    :return: Dropout
    """
    return Lambda(lambda x: K.dropout(x, level=rate))
