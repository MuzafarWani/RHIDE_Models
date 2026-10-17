# src/models/split_graph_vgg16.py
import tensorflow as tf
from tensorflow.keras.models import Model

# Mapping from split name → layer name in VGG16
# These names must exactly match layer.name in the Keras VGG16 model
VGG16_SPLITS = {
    "block1_pool": "block1_pool",
    "block2_conv1": "block2_conv1",
    "block3_conv1": "block3_conv1",
    "block3_conv2": "block3_conv2",
    "block3_conv3": "block3_conv3",
    "block4_conv3": "block4_conv3",
    "block5_conv2": "block5_conv2",
    "block5_conv3": "block5_conv3",
}

def create_vgg16_submodels(full_model: tf.keras.Model, beta_name: str):
    """
    Given a full VGG16 model and a split point β (beta_name),
    return two submodels:

      - device_model: from input up to β
      - edge_model:   from β output to final output

    This implements the semantic meaning of "splitting at β".
    """

    if beta_name not in VGG16_SPLITS:
        raise ValueError(f"Unknown VGG16 split beta: {beta_name}")

    cut_layer_name = VGG16_SPLITS[beta_name]
    cut_layer = full_model.get_layer(cut_layer_name)

    # --------------------
    # Device-side model
    # --------------------
    # Runs from model input up to the chosen split layer
    device_model = Model(
        inputs=full_model.input,
        outputs=cut_layer.output,
        name=f"device_until_{beta_name}",
    )

    # --------------------
    # Edge-side model
    # --------------------
    # Takes the activation tensor produced by device_model
    # and runs the remaining layers of the network
    edge_input = tf.keras.Input(shape=cut_layer.output.shape[1:])
    x = edge_input

    start = False
    for layer in full_model.layers:
        if layer.name == cut_layer_name:
            start = True
            continue
        if start:
            x = layer(x)

    edge_model = Model(
        inputs=edge_input,
        outputs=x,
        name=f"edge_from_{beta_name}",
    )

    # --------------------
    # Copy weights from the full model
    # --------------------
    for l in edge_model.layers:
        try:
            l.set_weights(full_model.get_layer(l.name).get_weights())
        except Exception:
            # Some layers (e.g., InputLayer) have no weights
            pass

    return device_model, edge_model

