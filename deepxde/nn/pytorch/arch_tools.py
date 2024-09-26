def calculate_cnn_output_sizes(image_size, convolutional_layers):
    """
    Calculates the image sizes at each step of the CNN.

    Args:
        image_size (tuple): Input image dimensions in the format (channel, height, width).
        convolutional_layers (list): List of dictionaries specifying the parameters of each convolutional layer.

    Returns:
        list: List of tuples representing the image size after each layer.
    """
    sizes = [image_size]
    height, width = image_size[1], image_size[2]
    
    for layer in convolutional_layers:
        # Calculate dimensions after the convolutional layer
        height = (height + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1
        width = (width + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"] + 1
        
        # Calculate dimensions after the pooling layer
        height = (height + 2 * layer["pool_padding"] - layer["pool_kernel_size"]) // layer["pool_stride"] + 1
        width = (width + 2 * layer["pool_padding"] - layer["pool_kernel_size"]) // layer["pool_stride"] + 1
        
        sizes.append((layer["out_channels"], height, width))
    
    return sizes
