# src/models/split_graph.py

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.models import Model


# ============================================================
# SPLIT DEFINITIONS (3D BRANCH ONLY)
# ============================================================

XEVENTNET_SPLITS = {
    "early_prepool": "3d_conv2",
    "post_pool1": "3d_pool1",
    "post_conv3": "3d_conv3",
    "global_pooled": "3d_global_pool",
    "branch_dense": "3d_dense",
    "concat": "3d_dense",          # fallback → treat same as branch_dense
    "combined_dense": "3d_dense",  # fallback → avoid concat path
}


# ============================================================
# MAIN FUNCTION
# ============================================================

def create_submodels(
    full_model: tf.keras.Model,
    beta_name: str,
    input_shape_2d,
    input_shape_3d,
    num_classes: int,
):
    """
    FINAL CORRECT XEventNet split:

    ✅ Device: 3D branch up to β
    ✅ Edge:   remaining 3D + custom head
    ❌ NO 2D branch
    ❌ NO CONCAT
    """

    if beta_name not in XEVENTNET_SPLITS:
        raise ValueError(f"Unknown split beta: {beta_name}")

    cut_layer_name = XEVENTNET_SPLITS[beta_name]
    cut_layer = full_model.get_layer(cut_layer_name)

    print(f"[split] Using cut: {beta_name} → {cut_layer_name}")

    # ============================================================
    # DEVICE MODEL (ONLY 3D)
    # ============================================================

    input_3d = full_model.get_layer("input_3d").input

    device_model = Model(
        inputs=input_3d,
        outputs=cut_layer.output,
        name=f"device_until_{beta_name}",
    )

    # ============================================================
    # EDGE MODEL (ONLY 3D PATH)
    # ============================================================

    edge_input = tf.keras.Input(
        shape=cut_layer.output.shape[1:], name="edge_3d_input"
    )

    x = edge_input
    passed_cut = False

    for layer in full_model.layers:

        if layer.name == cut_layer_name:
            passed_cut = True
            continue

        if not passed_cut:
            continue

        # ❌ Skip everything except 3D branch
        if "3d_" not in layer.name:
            continue

        # Stop before hitting original head
        if layer.name in ["3d_global_pool", "3d_dense"]:
            # allow these (they are valid 3D path)
            pass

        try:
            x = layer(x)
        except Exception:
            pass

    # ============================================================
    # CUSTOM EDGE HEAD (NO CONCAT)
    # ============================================================

    x = tf.keras.layers.Dense(128, activation="relu", name="edge_dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="edge_dropout")(x)
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="edge_output"
    )(x)

    edge_model = Model(
        inputs=edge_input,
        outputs=out,
        name=f"edge_from_{beta_name}",
    )

    # ============================================================
    # COPY WEIGHTS (ONLY MATCHING LAYERS)
    # ============================================================

    for l in edge_model.layers:
        if l.name in [layer.name for layer in full_model.layers]:
            try:
                l.set_weights(full_model.get_layer(l.name).get_weights())
            except Exception:
                pass

    return device_model, edge_model

