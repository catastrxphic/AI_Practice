import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import plotly.colors as pc


def simple_nn(input_size = 2, hidden_sizes=[2,3], output_size=2):

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

'''
    Each Layer has a different color when plotted
'''
# def graph_nn(layers):
#     fig = go.Figure()
#     # create neurons positions
#     neuron_positions = []

#     for i, layer_size in enumerate(layers):
#         x_positions = [i]*layer_size
#         y_positions = np.linspace(1,-1,layer_size)
#         neuron_positions.append(list(zip(x_positions, y_positions)))

#         # adding the neurons as scatter points
#         for i, layer in enumerate(neuron_positions):
#             x,y = zip(*layer)
#             fig.add_trace(go.Scatter(
#                 x=x,
#                 y=y,
#                 mode='markers',
#                 marker=dict(size=10, color='blue'),
#                 name=f'Layer {i + 1}'
#             ))

#         # adding connection of neuron in different layers
#         connection_colors = pc.qualitative.Plotly
#         for i in range(len(neuron_positions)-1):
#             current_layer = neuron_positions[i]
#             next_layer = neuron_positions[i+1]
#             # cicle through different colors of pc.qualitative.Plotly
#             color = connection_colors[i%len(connection_colors)]

#             for x1, y1 in current_layer:
#                 for x2, y2 in next_layer:
#                     fig.add_trace(go.Scatter(
#                         x=[x1,x2],
#                         y= [y1,y2],
#                         mode='lines',
#                         line=dict(width=1, color=color),
#                         showlegend=False
#                     ))
#     return fig

'''
    Each full-path has a different color 
'''
def graph_nn(layers):
    fig = go.Figure()

    # Possitions for neurons
    neuron_positions = []

    for i, layer_size in enumerate(layers):
        x_positions = [i]*layer_size
        y_positions = np.linspace(1,-1,layer_size)
        neuron_positions.append(list(zip(x_positions, y_positions)))

    # add neurons as scatter points
    for i, layer in enumerate(neuron_positions):
            x,y = zip(*layer)
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(size=10, color='blue'),
                name=f'Layer {i + 1}'
            ))
    
    # Create colors for each full-path
    total_paths = np.prod(layers[1:])
    path_colors = pc.qualitative.Plotly * (total_paths//len(pc.qualitative.Plotly)+1)

    # adding connections for full-paths (helper function)
    def plot_path(current_layer, current_neuron_index, color, path=[]):
        path = path + [neuron_positions[current_layer][current_neuron_index]]
        if current_layer == len(neuron_positions)-1:
            # draw entire path
            x,y = zip(*path)
            fig.add_trace(go.Scatter(
                x = x, 
                y = y,
                mode = 'lines',
                line= dict(width = 1, color = color),
                showlegend= False
            ))
        else:
            # go to the next layer
            if current_layer + 1 < len(neuron_positions):
                for next_neuron_index in range(len(neuron_positions[current_layer+1])):
                    plot_path(current_layer+1, next_neuron_index, color, path)

    # iterate through all input neurons
    color_index = 0
    for input_neuron_index in range(len(neuron_positions[0])):
        for first_hidden_neuron_index in range(len(neuron_positions[1])):
            if color_index < len(path_colors):
                plot_path(0, input_neuron_index, path_colors[color_index])
                color_index += 1

    return fig