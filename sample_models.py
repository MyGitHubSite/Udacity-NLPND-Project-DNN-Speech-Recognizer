from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
        TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    layer=input_data
    # Add recurrent layer
    layer = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(layer)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, recurrent_dropout, dropout, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))   
    layer=input_data
    # Add recurrent layer        
    layer = GRU(units, activation=activation, return_sequences=True, implementation=2, recurrent_dropout=recurrent_dropout, dropout=dropout, name='rnn')(layer)
    # TODO: Add batch normalization 
    layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recurrent_dropout, dropout, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input    s 
    input_data = Input(name='the_input', shape=(None, input_dim))   
    layer=input_data
    # Add convolutional layer
    layer = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='cnn')(layer)  
    # Add batch normalization
    layer = BatchNormalization()(layer)
    # Add a recurrent layer
    layer = GRU(units, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=recurrent_dropout, dropout=dropout, name='rnn')(layer)
    # TODO: Add batch normalization    
    layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}

    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1

    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recurrent_dropout, dropout, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    layer = input_data
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        layer = GRU(units, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=recurrent_dropout, dropout=dropout, name='rnn'+str(i))(layer)
        # Add batch normalization 
        layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer    
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, recurrent_dropout, dropout, recur_layers, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input    
    input_data = Input(name='the_input', shape=(None, input_dim))
    layer=input_data
    # TODO: Add bidirectional recurrent layer
    for i in range(recur_layers):
        layer = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=recurrent_dropout, dropout=dropout, name='rnn'+str(i)))(layer)
        # Add Batch Normalization
        layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recurrent_dropout, dropout, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input    
    input_data = Input(name='the_input', shape=(None, input_dim))
    layer=input_data
    # TODO: Specify the layers in your network
    layer = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(layer)
    # Add batch normalization
    layer = BatchNormalization()(layer)
    # Add recurrent layers
    for i in range(recur_layers):
       	layer = GRU(units, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=recurrent_dropout, dropout=dropout, name='rnn'+str(i))(layer)
        # Add batch normalization
       	layer = BatchNormalization()(layer)
    # Add a TimeDistributed(Dense(output_dim)) layer       	
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer    
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model_alt(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recurrent_dropout, dropout, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input    
    input_data = Input(name='the_input', shape=(None, input_dim))
    layer=input_data
    # TODO: Specify the layers in your network
    layer = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(layer)
    # Add batch normalization
    layer = BatchNormalization()(layer)
    # Add recurrent layers
    for i in range(recur_layers):
        layer = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, recurrent_dropout=recurrent_dropout, dropout=dropout, name='rnn'+str(i)))(layer)
        # Add batch normalization        
        layer = BatchNormalization()(layer)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer    
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
