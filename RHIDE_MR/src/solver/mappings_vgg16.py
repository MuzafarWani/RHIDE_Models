# src/solver/mappings_vgg16.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Beta:
    name: str

def vgg16_betas():
    """
    Must match configs_vgg16/split.yaml EXACTLY in order.

    These β correspond to valid layer names in VGG16 and to the
    entries in:
      - split.yaml.allowed
      - energy_latency.yaml per_beta arrays
      - split_graph_vgg16.py (VGG16_SPLITS)

    Order is shallow → deep.
    """
    return [
        Beta("block1_pool"),    # after first pooling
        Beta("block2_conv1"),   # early block2
        Beta("block3_conv1"),   # early block3
        Beta("block3_conv2"),   # mid block3
        Beta("block3_conv3"),   # late block3
        Beta("block4_conv3"),   # end of block4
        Beta("block5_conv2"),   # mid block5
        Beta("block5_conv3"),   # end of block5
    ]

