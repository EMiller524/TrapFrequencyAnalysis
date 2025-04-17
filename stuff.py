import math

'''
Ignore
'''
# n=5
# print(3*n)
# print(math.comb(3 * n, 3, repetition=True) + math.comb(3 * n, 4, repetition=True))

# print((-6.22298564e-6 + 8.37724820e-6)/1)


from math import comb, factorial

# parameters you can adjust:
n = 20
m = 80
# denominator is (m-n)! which for m=80, n=20 is 60!
den = factorial(m - n)

total = 0
for k in range(n + 1):
    term = (-1) ** k * comb(n, k) * factorial(m - k) / den
    total += term

# If you want an integer result:
total = total

print(total)
print(factorial(m) / (den))