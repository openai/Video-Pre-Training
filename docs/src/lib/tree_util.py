# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied this from jax, made it self-contained
# Currently just used for improved_checkpoint

import collections
import functools
import itertools as it
from collections.abc import Collection
from typing import Dict, List, Optional


def unzip2(xys):
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


def partial(fun, *args, **kwargs):
    wrapped = functools.partial(fun, *args, **kwargs)
    functools.update_wrapper(wrapped, fun)
    wrapped._bound_args = args  # pylint: disable=protected-access
    return wrapped


def safe_zip(*args: Collection) -> List[tuple]:
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, "length mismatch: {}".format(list(map(len, args)))
    return list(zip(*args))


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, "length mismatch: {}".format(list(map(len, args)))
    return list(map(f, *args))


def tree_map(f, tree, treat_as_leaves: Optional[List] = None):
    """Map a function over a pytree to produce a new pytree.

    Args:
      f: function to be applied at each leaf.
      tree: a pytree to be mapped over.

    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x)` where `x` is the value at the corresponding leaf in
      `tree`.
    """
    if treat_as_leaves is None:
        treat_as_leaves = []
    node_type = node_types.get(type(tree))
    if node_type and type(tree) not in treat_as_leaves:
        children, node_spec = node_type.to_iterable(tree)
        new_children = [tree_map(f, child, treat_as_leaves) for child in children]
        return node_type.from_iterable(node_spec, new_children)
    else:
        return f(tree)


def tree_multimap(f, tree, *rest, treat_as_leaves: Optional[List] = None):
    """Map a multi-input function over pytree args to produce a new pytree.

    Args:
      f: function that takes `1 + len(rest)` arguments, to be applied at the
        corresponding leaves of the pytrees.
      tree: a pytree to be mapped over, with each leaf providing the first
        positional argument to `f`.
      *rest: a tuple of pytrees, each with the same structure as `tree`.

    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x, *xs)` where `x` is the value at the corresponding leaf
      in `tree` and `xs` is the tuple of values at corresponding leaves in `rest`.
    """

    if treat_as_leaves is None:
        treat_as_leaves = []
    node_type = node_types.get(type(tree))
    if node_type and type(tree) not in treat_as_leaves:
        children, node_spec = node_type.to_iterable(tree)
        all_children = [children]
        for other_tree in rest:
            other_children, other_node_data = node_type.to_iterable(other_tree)
            if other_node_data != node_spec:
                raise TypeError("Mismatch: {} != {}".format(other_node_data, node_spec))
            all_children.append(other_children)

        new_children = [tree_multimap(f, *xs, treat_as_leaves=treat_as_leaves) for xs in zip(*all_children)]
        return node_type.from_iterable(node_spec, new_children)
    else:
        return f(tree, *rest)


def prefix_multimap(f, treedef, tree, *rest):
    """Like tree_multimap but only maps down through a tree prefix."""
    if isinstance(treedef, PyLeaf):
        return f(tree, *rest)
    else:
        node_type = node_types.get(type(tree))
        if node_type != treedef.node_type:
            raise TypeError("Mismatch: {} != {}".format(treedef.node_type, node_type))
        children, node_data = node_type.to_iterable(tree)
        if node_data != treedef.node_data:
            raise TypeError("Mismatch: {} != {}".format(treedef.node_data, node_data))
        all_children = [children]
        for other_tree in rest:
            other_children, other_node_data = node_type.to_iterable(other_tree)
            if other_node_data != node_data:
                raise TypeError("Mismatch: {} != {}".format(other_node_data, node_data))
            all_children.append(other_children)
        all_children = zip(*all_children)

        new_children = [prefix_multimap(f, td, *xs) for td, xs in zip(treedef.children, all_children)]
        return node_type.from_iterable(node_data, new_children)


def walk_pytree(f_node, f_leaf, tree, treat_as_leaves: Optional[List] = None):
    node_type = node_types.get(type(tree))
    if treat_as_leaves is None:
        treat_as_leaves = []

    if node_type and type(tree) not in treat_as_leaves:
        children, node_spec = node_type.to_iterable(tree)
        proc_children, child_specs = unzip2([walk_pytree(f_node, f_leaf, child, treat_as_leaves) for child in children])
        tree_def = PyTreeDef(node_type, node_spec, child_specs)
        return f_node(proc_children), tree_def
    else:
        return f_leaf(tree), PyLeaf()


def build_tree(treedef, xs):
    if isinstance(treedef, PyLeaf):
        return xs
    else:
        # We use 'iter' for clearer error messages
        children = safe_map(build_tree, iter(treedef.children), iter(xs))
        return treedef.node_type.from_iterable(treedef.node_data, children)


def _tree_unflatten(xs, treedef):
    if isinstance(treedef, PyLeaf):
        return next(xs)
    else:
        children = safe_map(partial(_tree_unflatten, xs), treedef.children)
        return treedef.node_type.from_iterable(treedef.node_data, children)


def _num_leaves(treedef):
    return 1 if isinstance(treedef, PyLeaf) else sum(safe_map(_num_leaves, treedef.children))


def _nested_treedef(inner, outer):
    # just used in tree_transpose error checking
    if isinstance(outer, PyLeaf):
        return inner
    else:
        children = safe_map(partial(_nested_treedef, inner), outer.children)
        return PyTreeDef(outer.node_type, outer.node_data, tuple(children))


class PyTreeDef(object):
    def __init__(self, node_type, node_data, children):
        self.node_type = node_type
        self.node_data = node_data
        self.children = children

    def __repr__(self):
        if self.node_data is None:
            data_repr = ""
        else:
            data_repr = "[{}]".format(self.node_data)

        return "PyTree({}{}, [{}])".format(self.node_type.name, data_repr, ",".join(safe_map(repr, self.children)))

    def __hash__(self):
        return hash((self.node_type, self.node_data, tuple(self.children)))

    def __eq__(self, other):
        if isinstance(other, PyLeaf):
            return False
        else:
            return self.node_type == other.node_type and self.node_data == other.node_data and self.children == other.children

    def __ne__(self, other):
        return not self == other


class PyLeaf(object):
    def __repr__(self):
        return "*"

    def __eq__(self, other):
        return isinstance(other, PyLeaf)


class NodeType(object):
    def __init__(self, name, to_iterable, from_iterable):
        self.name = name
        self.to_iterable = to_iterable
        self.from_iterable = from_iterable


node_types: Dict[type, NodeType] = {}


def register_pytree_node(py_type, to_iterable, from_iterable):
    assert py_type not in node_types
    node_types[py_type] = NodeType(str(py_type), to_iterable, from_iterable)


def tuple_to_iterable(xs):
    return xs, None


def tuple_from_iterable(_keys, xs):
    return tuple(xs)


def list_to_iterable(xs):
    return tuple(xs), None


def list_from_iterable(_keys, xs):
    return list(xs)


def dict_to_iterable(xs):
    keys = tuple(sorted(xs.keys()))
    return tuple(map(xs.get, keys)), keys


def dict_from_iterable(keys, xs):
    return dict(safe_zip(keys, xs))


def ordered_dict_from_iterable(keys, xs):
    return collections.OrderedDict(safe_zip(keys, xs))


def default_dict_to_iterable(xs):
    return (tuple(xs.values()), (xs.default_factory, tuple(xs.keys())))


def default_dict_from_iterable(keys, xs):
    return collections.defaultdict(keys[0], safe_zip(keys[1], xs))


def none_to_iterable(_xs):
    return (), None


def none_from_iterable(_keys, _xs):
    return None


register_pytree_node(tuple, tuple_to_iterable, tuple_from_iterable)
register_pytree_node(list, list_to_iterable, list_from_iterable)
register_pytree_node(dict, dict_to_iterable, dict_from_iterable)
register_pytree_node(collections.OrderedDict, dict_to_iterable, ordered_dict_from_iterable)
register_pytree_node(collections.defaultdict, default_dict_to_iterable, default_dict_from_iterable)
register_pytree_node(type(None), none_to_iterable, none_from_iterable)
