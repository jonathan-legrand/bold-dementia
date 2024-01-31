from functools import reduce
import sys
def itersize(iterable):
    return reduce(
        lambda x, y: x+y,
        map(sys.getsizeof, iterable)
    )