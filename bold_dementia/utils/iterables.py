from functools import reduce
import sys
from itertools import filterfalse
from operator import add

def unique(iterable):
    """
    List unique elements, preserving order. Remember all elements ever seen.
    Siplified version of unique_everseen from itertools recipes.
    """
    seen = set()
    for element in filterfalse(seen.__contains__, iterable):
        seen.add(element)
        yield element

def itersize(iterable):
    return reduce(
        add,
        map(sys.getsizeof, iterable)
    )