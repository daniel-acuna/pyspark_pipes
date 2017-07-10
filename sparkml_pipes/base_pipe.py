import functools


class BasePipe:
    """
    Represents a base infix pipe. Code inspired by https://github.com/JulienPalard/Pipe/blob/master/base_pipe.py
    """
    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __ror__(self, other):
        return self.function(other)

    def __call__(self, *args, **kwargs):
        return BasePipe(lambda x: self.function(x, *args, **kwargs))


@BasePipe
def take(iterable, qte):
    "Yield qte of elements in the given iterable."
    for item in iterable:
        if qte > 0:
            qte -= 1
            yield item
        else:
            return