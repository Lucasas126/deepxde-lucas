import torch
import torch.nn as nn
import torch.nn.functional as F
from deepxde.nn import NN
from deepxde.nn import activations
from deepxde.nn import initializers

class CNN(NN):
    """
    CNN arquitecture designed for use with deeponet:
    Composed of N convolutional_layers descfribed by the dictionary, which specifies kernel sizes, paddings and output channels.
    Finally, a fully connected network with different activations can be built.
    
    Args:
        image_size (tuple): Dimensiones de la imagen de entrada en formato (canal, alto, ancho).
        output_size (int): Tamaño de la salida deseada.
        convolutional_layers (list): Lista de diccionarios que especifican los parámetros de cada capa convolucional.
        layer_sizes (list): Lista que especifica los tamaños de las capas ocultas después de la parte convolucional.
        activation (str or list): Función de activación a aplicar a las capas ocultas y de salida.
            Puede ser una lista de funciones para especificar activaciones distintas para cada capa oculta.
        kernel_initializer (str): Método de inicialización de los pesos de las capas convolucionales.
        dropout_rate (float, optional): Tasa de dropout aplicada después de cada capa convolucional y lineal. Por defecto es 0.
    """
    
    def __init__(self, image_size, output_size, convolutional_layers, layer_sizes, activation, kernel_initializer, dropout_rate=0):
        super().__init__()

        self.image_size = image_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        # Define convolutional and pooling layers
        self.conv_layers = nn.ModuleList()
        current_channels = image_size[0]

        for conv_layer in convolutional_layers:
            conv = nn.Conv2d(
                in_channels=current_channels,
                out_channels=conv_layer["out_channels"],
                kernel_size=conv_layer["kernel_size"],
                stride=conv_layer["stride"],
                padding=conv_layer["padding"]
            )
            pool = nn.MaxPool2d(
                kernel_size=conv_layer["pool_kernel_size"],
                stride=conv_layer["pool_stride"],
                padding=conv_layer["pool_padding"]
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(pool)
            self.conv_layers.append(nn.Dropout2d(p=self.dropout_rate))
            initializer(conv.weight)
            initializer_zero(conv.bias)
            current_channels = conv_layer["out_channels"]

        conv_output_size = self._calculate_conv_output_size(image_size, convolutional_layers)

        self.linears = nn.ModuleList()
        flattened_size = conv_output_size[0] * conv_output_size[1] * current_channels
        
        layer_sizes = [flattened_size] + layer_sizes + [output_size]

        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Dropout(p=self.dropout_rate))
            self.linears.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32)
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

        if isinstance(activation, list):
            if not (len(layer_sizes)-1) == len(activation):
                raise ValueError(
                    f"Total number of activation functions do not match with sum of hidden layers and output layer!\nNº of activation function required: {len(layer_sizes)-1}"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
            
    def _calculate_conv_output_size(self, image_size, conv_layers):
        height, width = image_size[1], image_size[2]
        for conv_layer in conv_layers:
            height = (height + 2 * conv_layer["padding"] - conv_layer["kernel_size"]) // conv_layer["stride"] + 1
            width = (width + 2 * conv_layer["padding"] - conv_layer["kernel_size"]) // conv_layer["stride"] + 1
            height = (height + 2 * conv_layer["pool_padding"] - conv_layer["pool_kernel_size"]) // conv_layer["pool_stride"] + 1
            width = (width + 2 * conv_layer["pool_padding"] - conv_layer["pool_kernel_size"]) // conv_layer["pool_stride"] + 1
        return (height, width)

    def forward(self, inputs):
        x = inputs
        
        if self._input_transform is not None:
            x = self._input_transform(x)

        # CNN LAYERS
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)
            # elif isinstance(layer, nn.MaxPool2d):
            #     x = F.relu(x)
            # elif isinstance(layer, nn.Dropout2d):
            #     x = layer(x)

        # LINEAR LAYERS
        x = torch.flatten(x, 1)

        # Apply linear layers with dropout
        for j in range(len(self.linears)):
            if isinstance(self.linears[j], nn.Dropout):
                x = self.linears[j](x)
            else:
                x = (
                    self.activation[j//2](self.linears[j](x))
                    if isinstance(self.activation, list)
                    else self.activation(self.linears[j](x))
                )

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return x