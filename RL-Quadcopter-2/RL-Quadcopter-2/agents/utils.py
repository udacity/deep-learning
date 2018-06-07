import numpy as np
import tensorflow as tf


def scope_variables_mapping(from_scope_name, to_scope_name):
    vars_from = {var.name[len(from_scope_name) + 1:]: var for var in tf.trainable_variables(self,from_scope_name)}
    vars_to = {var.name[len(to_scope_name) + 1:]: var for var in tf.trainable_variables(to_scope_name)}

    mapped_names = set(vars_from.keys()).intersection(set(vars_to.keys()))

    return [[vars_from[name], vars_to[name]] for name in mapped_names]