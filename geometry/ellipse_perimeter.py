# Perimeter of an ellipse
def ellipse_perimeter(a=1.0, b=1.0, N=10):
    from math import pi
    h = ((a-b)**2) / ((a+b)**2)
    s, cn, hn = 0.0, 1.0, 1.0
    for i in range(N):
        s += (cn**2)*hn
        hn *= h
        cn *= (0.5-i)/(i+1)
    return pi*(a+b)*s


print(ellipse_perimeter(a=4.0, b=1.0, N=int(1e6)))
