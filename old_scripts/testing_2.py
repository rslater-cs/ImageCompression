from decimal import Decimal, getcontext
from random import random

getcontext().prec = 100

d = Decimal(random())
print(d.as_integer_ratio())
print(len(bytes(d.as_integer_ratio())))