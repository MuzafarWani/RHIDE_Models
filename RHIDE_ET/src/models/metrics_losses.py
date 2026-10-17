# src/models/metrics_losses.py
from __future__ import annotations

from typing import Optional, Dict

import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    """
    Simple macro-F1 for multi-class classification.
    """
    def __init__(self, num_classes: int, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        for c in range(self.num_classes):
            yc = tf.equal(y_true, c)
            pc = tf.equal(y_pred, c)
            tp = tf.reduce_sum(tf.cast(tf.logical_and(yc, pc), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(yc), pc), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(yc, tf.logical_not(pc)), tf.float32))
            self.tp[c].assign_add(tp)
            self.fp[c].assign_add(fp)
            self.fn[c].assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_states(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


def make_class_weighted_loss(class_weights: Dict[int, float]):
    """
    Returns a sparse-categorical-crossentropy loss that applies per-class weights.
    """
    cw = tf.constant([class_weights.get(i, 1.0) for i in range(max(class_weights)+1)], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        weights = tf.gather(cw, y_true)
        scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(scce * weights)
    return loss_fn


def focal_loss(gamma: float = 2.0, alpha: Optional[Dict[int, float]] = None):
    """
    Multi-class focal loss with optional per-class alpha.
    """
    alpha_vec = None
    if alpha is not None:
        alpha_vec = tf.constant([alpha.get(i, 1.0) for i in range(max(alpha)+1)], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # one-hot for focal formulation
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        fl = (1.0 - pt) ** gamma * tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred)
        if alpha_vec is not None:
            weights = tf.gather(alpha_vec, y_true)
            fl = fl * weights
        return tf.reduce_mean(fl)
    return loss_fn

