from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import network as nn


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Neural Netwrok Visualizer"),
    dcc.Graph(id='Neural-Network-Visualization'),
    dcc.Slider(1,5,step=1,value=2, id='hidden-layer-slider')
])




@app.callback(
    Output("Neural-Network-Visualization", "figure"),
    Input("hidden-layer-slider","value")
)

def update_network(hidden_layers):
    layers = [2] + [hidden_layers] * 2 + [1]
    return nn.graph_nn(layers)

if __name__ == '__main__':
    app.run_server(debug=True)
