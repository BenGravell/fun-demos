"""Computational verification of the Collatz conjecture up to integer N using memoization"""
from time import time
from collections import defaultdict


def rule(n):
    return 3*n + 1 if n % 2 else n//2


# Initialize the memo dictionary
d = defaultdict(bool)
d[1] = True
# Max number to check the Collatz conjecture on
N = 10000000

# Start checking
print("Verifying Collatz conjecture for all integers from 1 to %d..." % N, end='')
t_start = time()
max_iters = 0
for i in range(2, N+1):
    n = i
    s = [n]
    iters = 0
    while not d[n]:
        n = rule(n)
        s.append(n)
        iters += 1
    for k in s:
        d[k] = True
    max_iters = max(iters, max_iters)
print("done!")
# Post-analysis
t_end = time()
print("Elapsed time: %.3f seconds" % (t_end-t_start))
print("Max number of iterations-to-convergence encountered = %d" % max_iters)
