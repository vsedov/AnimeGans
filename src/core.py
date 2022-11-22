class HC:
    """Core CLS for constants"""

    def __init__(self):
        ...


hc = HC()


def hc_register(f):
    """Another potential register, although not required."""
    hc.__dict__[f.__name__] = f
    return f


def hc_deco(f):
    """
    Register Decorators as a constant
    -- This needs to be called first
    """

    def wrapped(*args, **kwargs):
        hc.__dict__[f.__name__] = f
        return f(*args, **kwargs)

    return wrapped


def hc_register_const(name, value):
    """Register a constant using this function, fixed values"""
    hc.__dict__[name] = value

    return value


class HelperFunctions:
    """cls for pure functions no constants, globally defined constant functions or classes"""

    def __init__(self):
        ...


hp = HelperFunctions()


def hp_register(f):
    """Register a global function.
    -- registered through a decorator but does not need to be called.
    """
    hp.__dict__[f.__name__] = f
    return f
