import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

def simple_nn(input_size = 2, hidden_sizes=[2,3], output_size=1):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_size)))
    for hidden_size in hidden_sizes:
        model.add(tf.keras.layers.Dense(hidden_size, activation="relu"))
    model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))

    return model

def data_generation(samples = 100):
    x = np.random.rand(samples, 2)
    y = (x[:, 0] + x[:, 1] > 1).astype(int)
    return x,y

def graph_nn(layers):
    fig = go.Figure()
    y_position = np.linspace(-1,1,len(layers))

    for i, layer_size in enumerate(layers):
        x_positions = np.linspace(-1, 1, layer_size)
        fig.add_trace(go.Scatter(
            x=[i] * len(x_positions),
            y=x_positions,
            mode='markers',
            marker=dict(size=10),
            name=f'Layer {i + 1}'
        ))
    return fig