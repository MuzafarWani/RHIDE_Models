# src/models/split_graph.py  for XEvenetNet
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense,
    Conv3D, MaxPooling3D, GlobalAveragePooling3D, Concatenate, Dropout, Layer
)

"""
Split-graph utilities.

We expose:
- get_allowed_cuts(full_model, split_cfg) -> List[CutSpec]
- create_submodels(full_model, beta_name, input_shapes) -> (device_net, edge_net)

Design:
- We rebuild the two submodels with the SAME layer names so we can copy weights
  over by name from the full model.
- We support several paired cut points (pre-pool, post-pool, etc.) and joint cuts.
"""

@dataclass(frozen=True)
class CutSpec:
    name: str
    type: str  # "paired" (needs 2d & 3d) or "joint" (single tensor)
    two_d_anchor: Optional[str] = None
    three_d_anchor: Optional[str] = None
    joint_anchor: Optional[str] = None  # for "concat" / "combined_dense" / "output"


# Canonical cut table
DEFAULT_CUTS: List[CutSpec] = [
    CutSpec("early_prepool",        "paired", two_d_anchor="2d_conv2",        three_d_anchor="3d_conv2"),
    CutSpec("post_pool1",           "paired", two_d_anchor="2d_pool1",        three_d_anchor="3d_pool1"),
    CutSpec("post_conv3",           "paired", two_d_anchor="2d_conv3",        three_d_anchor="3d_conv3"),
    CutSpec("global_pooled",        "paired", two_d_anchor="2d_global_pool",  three_d_anchor="3d_global_pool"),
    CutSpec("branch_dense",         "paired", two_d_anchor="2d_dense",        three_d_anchor="3d_dense"),
    CutSpec("concat",               "joint",  joint_anchor="concat"),
    CutSpec("combined_dense",       "joint",  joint_anchor="combined_dense"),
    CutSpec("output",               "joint",  joint_anchor="output"),  # β = L (fully on device)
]


def get_allowed_cuts(full_model: Model, split_cfg: Dict) -> List[CutSpec]:
    """
    Filters DEFAULT_CUTS by split_cfg["allowed"] if provided.
    """
    allowed_names = split_cfg.get("allowed", None)
    cuts = [c for c in DEFAULT_CUTS if (allowed_names is None or c.name in allowed_names)]
    if not cuts:
        raise ValueError("No allowed cut points after filtering; check split_cfg['allowed'].")
    return cuts


def _build_2d_branch_until(anchor: str, inp):
    x = inp
    # Mirror architecture to anchor
    if anchor in ("2d_conv1", "2d_conv2", "2d_pool1", "2d_conv3", "2d_global_pool", "2d_dense"):
        x = Conv2D(32, (3,3), activation="relu", padding="same", name="2d_conv1")(x)
        if anchor == "2d_conv1": return x
        x = Conv2D(64, (3,3), activation="relu", padding="same", name="2d_conv2")(x)
        if anchor == "2d_conv2": return x
        x = MaxPooling2D((2,2), name="2d_pool1")(x)
        if anchor == "2d_pool1": return x
        x = Conv2D(128, (3,3), activation="relu", padding="same", name="2d_conv3")(x)
        if anchor == "2d_conv3": return x
        x = GlobalMaxPooling2D(name="2d_global_pool")(x)
        if anchor == "2d_global_pool": return x
        x = Dense(128, activation="relu", name="2d_dense")(x)
        if anchor == "2d_dense": return x
    raise ValueError(f"Unknown 2D anchor: {anchor}")


def _build_3d_branch_until(anchor: str, inp):
    x = inp
    if anchor in ("3d_conv1", "3d_conv2", "3d_pool1", "3d_conv3", "3d_global_pool", "3d_dense"):
        x = Conv3D(32, (3,3,3), activation="relu", padding="same", name="3d_conv1")(x)
        if anchor == "3d_conv1": return x
        x = Conv3D(64, (3,3,3), activation="relu", padding="same", name="3d_conv2")(x)
        if anchor == "3d_conv2": return x
        x = MaxPooling3D((2,2,2), name="3d_pool1")(x)
        if anchor == "3d_pool1": return x
        x = Conv3D(128, (3,3,3), activation="relu", padding="same", name="3d_conv3")(x)
        if anchor == "3d_conv3": return x
        x = GlobalAveragePooling3D(name="3d_global_pool")(x)
        if anchor == "3d_global_pool": return x
        x = Dense(128, activation="relu", name="3d_dense")(x)
        if anchor == "3d_dense": return x
    raise ValueError(f"Unknown 3D anchor: {anchor}")


def _build_2d_branch_from(anchor: str, inp):
    """
    Builds the remainder of the 2D branch AFTER anchor output.
    """
    x = inp
    order = ["2d_conv1","2d_conv2","2d_pool1","2d_conv3","2d_global_pool","2d_dense"]
    idx = order.index(anchor)
    for name in order[idx+1:]:
        if name == "2d_conv2":
            x = Conv2D(64, (3,3), activation="relu", padding="same", name="2d_conv2")(x)
        elif name == "2d_pool1":
            x = MaxPooling2D((2,2), name="2d_pool1")(x)
        elif name == "2d_conv3":
            x = Conv2D(128, (3,3), activation="relu", padding="same", name="2d_conv3")(x)
        elif name == "2d_global_pool":
            x = GlobalMaxPooling2D(name="2d_global_pool")(x)
        elif name == "2d_dense":
            x = Dense(128, activation="relu", name="2d_dense")(x)
    return x


def _build_3d_branch_from(anchor: str, inp):
    x = inp
    order = ["3d_conv1","3d_conv2","3d_pool1","3d_conv3","3d_global_pool","3d_dense"]
    idx = order.index(anchor)
    for name in order[idx+1:]:
        if name == "3d_conv2":
            x = Conv3D(64, (3,3,3), activation="relu", padding="same", name="3d_conv2")(x)
        elif name == "3d_pool1":
            x = MaxPooling3D((2,2,2), name="3d_pool1")(x)
        elif name == "3d_conv3":
            x = Conv3D(128, (3,3,3), activation="relu", padding="same", name="3d_conv3")(x)
        elif name == "3d_global_pool":
            x = GlobalAveragePooling3D(name="3d_global_pool")(x)
        elif name == "3d_dense":
            x = Dense(128, activation="relu", name="3d_dense")(x)
    return x


def _build_head_from(joint_anchor: str, inp):
    """
    Head after 'concat' or 'combined_dense'.
    """
    x = inp
    if joint_anchor == "concat":
        # remaining: combined_dense -> combined_dropout -> output
        x = Dense(128, activation="relu", name="combined_dense")(x)
        x = Dropout(0.5, name="combined_dropout")(x)
        x = Dense(tf.keras.backend.int_shape(x)[-1] or 128, name="__noop__")(x)  # placeholder to preserve graph
        # We can't infer num_classes from shape here; will attach 'output' in edge builder below
        return x
    elif joint_anchor == "combined_dense":
        # remaining: combined_dropout -> output
        x = Dropout(0.5, name="combined_dropout")(x)
        return x
    elif joint_anchor == "output":
        # nothing after output
        return x
    else:
        raise ValueError(f"Unknown joint anchor: {joint_anchor}")


def create_submodels(
    full_model: Model,
    beta_name: str,
    input_shape_2d: Tuple[int, int, int],
    input_shape_3d: Tuple[int, int, int, int],
    num_classes: int,
) -> Tuple[Model, Optional[Model]]:
    """
    Builds (device_net, edge_net) for a given cut name.
    - If beta_name == "output": edge_net is None (everything on device).
    - For paired cuts: device_net outputs [A2d, A3d]; edge_net accepts two inputs.
    - For joint cuts: device_net outputs a single tensor; edge_net accepts one input.
    """
    cut = next((c for c in DEFAULT_CUTS if c.name == beta_name), None)
    if cut is None:
        raise ValueError(f"Unknown cut name: {beta_name}")
    
    # Log the cut specification
    print(f"[train][split] Creating submodels for cut: {cut.name} (beta={beta_name})")

    # DEVICE NET
    inp2 = Input(shape=input_shape_2d, name="input_2d")
    inp3 = Input(shape=input_shape_3d, name="input_3d")

    if cut.type == "paired":
        print(f"[train][split] Paired cut: two_d_anchor={cut.two_d_anchor}, three_d_anchor={cut.three_d_anchor}")
        
        a2 = _build_2d_branch_until(cut.two_d_anchor, inp2)
        a3 = _build_3d_branch_until(cut.three_d_anchor, inp3)
        device_out = [a2, a3]
        device_net = Model(inputs=[inp2, inp3], outputs=device_out, name=f"device_net_{beta_name}")

        # EDGE NET: take these activations and complete the network
        ed_in2 = Input(shape=a2.shape[1:], name=f"edge_in_{cut.two_d_anchor}")
        ed_in3 = Input(shape=a3.shape[1:], name=f"edge_in_{cut.three_d_anchor}")
        print(f"[train][split] Edge net inputs: {ed_in2.name}, {ed_in3.name}")

        tail2 = _build_2d_branch_from(cut.two_d_anchor, ed_in2)
        tail3 = _build_3d_branch_from(cut.three_d_anchor, ed_in3)
        merged = Concatenate(name="concat")([tail2, tail3])
        x = Dense(128, activation="relu", name="combined_dense")(merged)
        x = Dropout(0.5, name="combined_dropout")(x)
        out = Dense(num_classes, activation="softmax", name="output")(x)
        edge_net = Model(inputs=[ed_in2, ed_in3], outputs=out, name=f"edge_net_{beta_name}")
        
        # Log the layers involved in the split
        print(f"[train][split] Split at layer {cut.two_d_anchor} (2D) and {cut.three_d_anchor} (3D)")
        print(f"[train][split] Edge network after split: {', '.join([l.name for l in edge_net.layers])}")

    elif cut.type == "joint":
        print(f"[train][split] Joint cut: joint_anchor={cut.joint_anchor}")
        
        # Build full branches to the joint anchor on device
        a2 = _build_2d_branch_until("2d_dense", inp2)
        a3 = _build_3d_branch_until("3d_dense", inp3)
        merged = Concatenate(name="concat")([a2, a3])

        if cut.joint_anchor == "concat":
            device_out = merged
            device_net = Model(inputs=[inp2, inp3], outputs=device_out, name=f"device_net_{beta_name}")
            ed_in = Input(shape=merged.shape[1:], name=f"edge_in_{cut.joint_anchor}")
            print(f"[train][split] Edge net input: {ed_in.name}")
            x = Dense(128, activation="relu", name="combined_dense")(ed_in)
            x = Dropout(0.5, name="combined_dropout")(x)
            out = Dense(num_classes, activation="softmax", name="output")(x)
            edge_net = Model(inputs=ed_in, outputs=out, name=f"edge_net_{beta_name}")

            # Log the layers involved in the joint split
            print(f"[train][split] Joint split at layer {cut.joint_anchor}")
            print(f"[train][split] Edge network after split: {', '.join([l.name for l in edge_net.layers])}")

        elif cut.joint_anchor == "combined_dense":
            x = Dense(128, activation="relu", name="combined_dense")(merged)
            device_out = x
            device_net = Model(inputs=[inp2, inp3], outputs=device_out, name=f"device_net_{beta_name}")
            ed_in = Input(shape=x.shape[1:], name=f"edge_in_{cut.joint_anchor}")
            print(f"[train][split] Edge net input: {ed_in.name}")
            x = Dropout(0.5, name="combined_dropout")(ed_in)
            out = Dense(num_classes, activation="softmax", name="output")(x)
            edge_net = Model(inputs=ed_in, outputs=out, name=f"edge_net_{beta_name}")

            # Log the layers involved in the joint split
            print(f"[train][split] Joint split at layer {cut.joint_anchor}")
            print(f"[train][split] Edge network after split: {', '.join([l.name for l in edge_net.layers])}")

        elif cut.joint_anchor == "output":
            # Fully on device
            x = Dense(128, activation="relu", name="combined_dense")(merged)
            x = Dropout(0.5, name="combined_dropout")(x)
            out = Dense(num_classes, activation="softmax", name="output")(x)
            device_net = Model(inputs=[inp2, inp3], outputs=out, name=f"device_net_{beta_name}")
            edge_net = None

            # Log that the output is fully on the device
            print(f"[train][split] Joint split at output layer (fully on device)")

        else:
            raise ValueError(f"Unsupported joint anchor: {cut.joint_anchor}")
        
        # Log the layers involved in the joint split
        print(f"[train][split] Edge network after split: {', '.join([l.name for l in edge_net.layers]) if edge_net else 'None (fully on device)'}")
    else:
        raise ValueError(f"Unsupported cut type: {cut.type}")

    return device_net, edge_net

