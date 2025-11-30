import random
import sys
import os

def generate_random_test(n, min_val=-1000, max_val=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    
    a = [random.randint(min_val, max_val) for _ in range(n)]
    return a

random_seq = generate_random_test(5000)
print(len(random_seq))
for i in random_seq:
    print(i)