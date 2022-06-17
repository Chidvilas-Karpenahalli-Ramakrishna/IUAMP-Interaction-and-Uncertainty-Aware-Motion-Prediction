# Required built-in modules:
# -----------------------------
import os

# Required project modules:
# -----------------------------
from model import supporting_functions

# Required TensorFlow modules:
# ----------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, \
    TimeDistributed, Reshape, Dense, MaxPooling2D, Conv2D, Dropout

# To prevent TensorFlow log outputs:
# ---------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Deterministic model:
# -----------------------
def deterministic_model(desired_shape):
    """
    A Deterministic model without any uncertainty quantification

    :param desired_shape: Shape of the input
    """
    model = Sequential()
    
    ## Encoder:
    ## ----------
    # A 2-layered Feature Extractor:
    # -----------------------------------
    model.add(TimeDistributed(Conv2D(filters=64, padding='same',
                                     kernel_size=(3, 3)),
                              input_shape=(None, desired_shape[0], desired_shape[1], 3)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='valid')))
    model.add(BatchNormalization())

    model.add(TimeDistributed(Conv2D(filters=32, padding='same',
                                     kernel_size=(2, 2))))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(BatchNormalization())

    # Spatio-temporal learning: 3 ConvLSTM layers
    # ---------------------------------------------------
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                         return_sequences=True, activation='relu', dropout=0.3))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same')))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=16, kernel_size=(2, 2), padding='same',
                         return_sequences=True, activation='relu', dropout=0.3))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=16, kernel_size=(2, 2), padding='same',
                         return_sequences=True, activation='relu', dropout=0.3))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    model.add(BatchNormalization())

    # Flatten the output:
    # ---------------------
    model.add(TimeDistributed(Flatten()))

    # Add Self-Attention layer:
    # -----------------------------
    model.add(supporting_functions.SeqSelfAttention(attention_width=4,
                                                    attention_activation='sigmoid',
                                                    attention_type='multiplicative'))

    ## Decoder:
    ## ----------
    """
    The output is sigmoid activated to keep the results between 0 and 1
    """
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(400, activation='sigmoid')))
    model.add(Reshape((-1, 20, 20, 1)))

    print("The model architecture is: ")
    model.summary()

    return model


# Uncertainty-aware model: Monte Carlo (MC) Dropout
# -----------------------------------------------------------
def uncertainty_aware_model(desired_shape):
    """
    An uncertainty-aware model which quantifies uncertainty using the MC Dropout
    method

    :param desired_shape: Shape of the input
    """
    model = Sequential()
    
    ## Encoder:
    ## ----------
    # A 2-layered Feature Extractor:
    # -----------------------------------
    model.add(TimeDistributed(Conv2D(filters=64, padding='same',
                                     kernel_size=(3, 3)),
                              input_shape=(None, desired_shape[0], desired_shape[1], 3)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='valid')))
    model.add(supporting_functions.permanent_dropout(0.3))
    model.add(BatchNormalization())

    model.add(TimeDistributed(Conv2D(filters=32, padding='same',
                                     kernel_size=(2, 2))))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid')))
    model.add(supporting_functions.permanent_dropout(0.3))
    model.add(BatchNormalization())

    # Spatio-temporal learning: 3 ConvLSTM layers
    # ---------------------------------------------------
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                         return_sequences=True, activation='relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same')))
    model.add(supporting_functions.permanent_dropout(0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=16, kernel_size=(2, 2), padding='same',
                         return_sequences=True, activation='relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    model.add(supporting_functions.permanent_dropout(0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=16, kernel_size=(2, 2), padding='same',
                         return_sequences=True, activation='relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    model.add(supporting_functions.permanent_dropout(0.3))
    model.add(BatchNormalization())

    # Flatten the output:
    # ---------------------
    model.add(TimeDistributed(Flatten()))

    # Add Self-Attention layer:
    # -----------------------------
    model.add(supporting_functions.SeqSelfAttention(attention_width=4,
                                                    attention_activation='sigmoid',
                                                    attention_type='multiplicative'))

    ## Decoder:
    ## ----------
    """
    The output layer is sigmoid activated to keep the outputs between 0 and 1
    """
    model.add(supporting_functions.permanent_dropout(0.4))
    model.add(TimeDistributed(Dense(400, activation='sigmoid')))
    model.add(Reshape((-1, 20, 20, 1)))

    print("The model architecture is: ")
    model.summary()

    return model


# Model options:
"""
If and when you define a new model above, please add it to the dictionary below.
These models can be chosen for training by defining in 'experimentation_parameters.json'.
"""
model_options = {"deterministic_model": deterministic_model,
                 "uncertainty_aware_model": uncertainty_aware_model}
