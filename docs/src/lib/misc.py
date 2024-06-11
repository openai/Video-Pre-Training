import numpy as np
import torch as th


def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


def safezip(*args):
    """
    Check that lengths of sequences are the same, then zip them
    """
    args = [list(a) for a in args]
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(zip(*args))


def transpose(x, before, after):
    """
    Usage: x_bca = transpose(x_abc, 'abc', 'bca')
    """
    assert sorted(before) == sorted(after), f"cannot transpose {before} to {after}"
    assert x.ndim == len(
        before
    ), f"before spec '{before}' has length {len(before)} but x has {x.ndim} dimensions: {tuple(x.shape)}"
    return x.permute(tuple(before.index(i) for i in after))


def transpose_undo(x, before, after, *, undo=None):
    """
    Usage:
    x_bca, undo = transpose_undo(x_abc, 'abc', 'bca')
    x_bca = fully_connected_layer(x_bca)
    x_abc = undo(x_bca)
    """
    return (
        transpose(x, before, after),
        compose_undo(undo, lambda x: transpose(x, before=after, after=before)),
    )


def compose_undo(u1, u2):
    assert u2 is not None
    if u1 is None:
        return u2

    def u(x):
        x = u2(x)
        x = u1(x)
        return x

    return u


NO_BIND = "__nobind"


def _parse_reshape_str(s, kind):
    assert kind in ("before", "after")
    result = []
    n_underscores = 0
    for i, part in enumerate(s.split(",")):
        part = part.strip()
        if part == "?" and kind == "before":
            result.append([f"__{i}"])
        elif part == "_":
            result.append([f"{NO_BIND}_{n_underscores}"])
            n_underscores += 1
        else:
            result.append([term.strip() for term in part.split("*")])
    return result


def _infer_part(part, concrete_dim, known, index, full_shape):
    if type(part) is int:
        return part
    assert isinstance(part, list), part
    lits = []
    syms = []
    for term in part:
        if type(term) is int:
            lits.append(term)
        elif type(term) is str:
            syms.append(term)
        else:
            raise TypeError(f"got {type(term)} but expected int or str")
    int_part = 1
    for x in lits:
        int_part *= x
    if len(syms) == 0:
        return int_part
    elif len(syms) == 1 and concrete_dim is not None:
        assert concrete_dim % int_part == 0, f"{concrete_dim} % {int_part} != 0 (at index {index}, full shape is {full_shape})"
        v = concrete_dim // int_part
        if syms[0] in known:
            assert (
                known[syms[0]] == v
            ), f"known value for {syms[0]} is {known[syms[0]]} but found value {v} at index {index} (full shape is {full_shape})"
        else:
            known[syms[0]] = v
        return concrete_dim
    else:
        for i in range(len(syms)):
            if syms[i] in known:
                syms[i] = known[syms[i]]
            else:
                try:
                    syms[i] = int(syms[i])
                except ValueError:
                    pass
        return lits + syms


def _infer_step(args):
    known, desc, shape = args
    new_known = known.copy()
    new_desc = desc.copy()
    for i in range(len(desc)):
        if shape is None:
            concrete_dim = None
        else:
            concrete_dim = shape[i]
        new_desc[i] = _infer_part(part=desc[i], concrete_dim=concrete_dim, known=new_known, index=i, full_shape=shape)
    return new_known, new_desc, shape


def _infer(known, desc, shape):
    if shape is not None:
        assert len(desc) == len(shape), f"desc has length {len(desc)} but shape has length {len(shape)} (shape={shape})"
    known, desc, shape = fixed_point(_infer_step, (known, desc, shape))
    return desc, known


def fixed_point(f, x, eq=None):
    if eq is None:
        eq = lambda a, b: a == b
    while True:
        new_x = f(x)
        if eq(x, new_x):
            return x
        else:
            x = new_x


def _infer_question_mark(x, total_product):
    try:
        question_mark_index = x.index(["?"])
    except ValueError:
        return x
    observed_product = 1
    for i in range(len(x)):
        if i != question_mark_index:
            assert type(x[i]) is int, f"when there is a question mark, there can be no other unknown values (full list: {x})"
            observed_product *= x[i]
    assert (
        observed_product and total_product % observed_product == 0
    ), f"{total_product} is not divisible by {observed_product}"
    value = total_product // observed_product
    x = x.copy()
    x[question_mark_index] = value
    return x


def _ground(x, known, infer_question_mark_with=None):
    x, known = _infer(known=known, desc=x, shape=None)
    if infer_question_mark_with:
        x = _infer_question_mark(x, infer_question_mark_with)
    for part in x:
        assert type(part) is int, f"cannot infer value of {part}"
    return x


def _handle_ellipsis(x, before, after):
    ell = ["..."]
    try:
        i = before.index(ell)
        l = len(x.shape) - len(before) + 1
        ellipsis_value = x.shape[i : i + l]
        ellipsis_value = list(ellipsis_value)
        before = before[:i] + ellipsis_value + before[i + 1 :]
    except ValueError:
        pass
    try:
        i = after.index(ell)
        after = after[:i] + ellipsis_value + after[i + 1 :]
    except ValueError:
        pass
    except UnboundLocalError as e:
        raise ValueError("there cannot be an ellipsis in 'after' unless there is an ellipsis in 'before'") from e
    return before, after


def reshape_undo(inp, before, after, *, undo=None, known=None, **kwargs):
    """
    Usage:
    x_Bhwse, undo = reshape_undo(
        x_bthwe,
        'b, t, ..., stride*e',
        'b*t, ..., stride, e',
        stride=7
    )
    x_Bhwse = do_some_stuff(x_Bhwse)
    x_bthwe = undo(x_Bhwse)

    It's necessary to pass known values as keywords only
    when they can't be inferred from the shape.

    (Eg. in the above example we needed to pass
    stride but not b, t, or e, since those can be determined from
    inp.shape once stride is known.)
    """
    if known:
        known = {**kwargs, **known}
    else:
        known = kwargs
    assert type(before) is type(after), f"{type(before)} != {type(after)}"
    assert isinstance(inp, (th.Tensor, np.ndarray)), f"require tensor or ndarray but got {type(inp)}"
    assert isinstance(before, (str, list)), f"require str or list but got {type(before)}"
    if isinstance(before, str):
        before = _parse_reshape_str(before, "before")
        after = _parse_reshape_str(after, "after")
        before, after = _handle_ellipsis(inp, before, after)
    before_saved, after_saved = before, after
    before, known = _infer(known=known, desc=before, shape=inp.shape)
    before = _ground(before, known, product(inp.shape))
    after = _ground(after, known, product(inp.shape))
    known = {k: v for k, v in known.items() if not k.startswith(NO_BIND)}
    assert tuple(inp.shape) == tuple(before), f"expected shape {before} but got shape {inp.shape}"
    assert product(inp.shape) == product(
        after
    ), f"cannot reshape {inp.shape} to {after} because the number of elements does not match"
    return (
        inp.reshape(after),
        compose_undo(undo, lambda inp: reshape(inp, after_saved, before_saved, known=known)),
    )


def reshape(*args, **kwargs):
    """
    Please see the documenation for reshape_undo.
    """
    x, _ = reshape_undo(*args, **kwargs)
    return x


def product(xs, one=1):
    result = one
    for x in xs:
        result = result * x
    return result


def exact_div(a, b):
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b
