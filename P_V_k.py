"""solves theorem 5.1"""
import sympy as sp
C = 2
x = sp.Symbol('x')
i = sp.Symbol('i', integer=True)
eps = sp.Symbol('eps')

if C == 0:
    f1 = 2/eps*x  # x in [0, eps/2]
    f2 = 1  # x in [eps/2, 1-eps/2]
    f3 = 2/eps*(1-x)  # x in [1-eps/2, 1]
    # a = 4*(4*pi**2*eps**2*i**2 - 4*pi**2*eps**2*i + 2*pi**2*eps**2*(4*i**2 - 4*i + 1)*cos(pi*eps*(i - 1/2))**2 + pi**2*eps**2 - 8*pi*eps*i*sin(pi*eps*(i - 1/2)) + 4*pi*eps*sin(pi*eps*(i - 1/2)) - 8*cos(pi*eps*(i - 1/2)) + 8)
    # b = (pi**4*eps**2*(4*i**2 - 4*i + 1)**2)
    # res = a/b ~1/i**2
elif C == 1:
    f1 = -2/eps**3 * x**3 + 3/eps**2*x**2
    f2 = 1
    f3 = -2/eps**3 * (1-x)**3 + 3/eps**2*(1-x)**2
    # (2199023255552.0*pi**6*eps**6*i**6*cos(pi*eps*(i - 1/2))**2 + 274877906944.0*pi**6*eps**6*i**6 - 6597069766656.0*pi**6*eps**6*i**5*cos(pi*eps*(i - 1/2))**2 - 824633720832.0*pi**6*eps**6*i**5 + 8246337208320.0*pi**6*eps**6*i**4*cos(pi*eps*(i - 1/2))**2 + 1030792151040.0*pi**6*eps**6*i**4 - 5497558138880.0*pi**6*eps**6*i**3*cos(pi*eps*(i - 1/2))**2 - 687194767360.0*pi**6*eps**6*i**3 + 2061584302080.0*pi**6*eps**6*i**2*cos(pi*eps*(i - 1/2))**2 + 257698037760.0*pi**6*eps**6*i**2 - 412316860416.0*pi**6*eps**6*i*cos(pi*eps*(i - 1/2))**2 - 51539607552.0*pi**6*eps**6*i + 34359738368.0*pi**6*eps**6*cos(pi*eps*(i - 1/2))**2 + 4294967296.0*pi**6*eps**6 + 1649267441664.0*pi**4*eps**4*i**4*cos(pi*eps*(i - 1/2)) + 618475290624.0*pi**4*eps**4*i**4 - 3298534883328.0*pi**4*eps**4*i**3*cos(pi*eps*(i - 1/2)) - 1236950581248.0*pi**4*eps**4*i**3 + 2473901162496.0*pi**4*eps**4*i**2*cos(pi*eps*(i - 1/2)) + 927712935936.0*pi**4*eps**4*i**2 - 824633720832.0*pi**4*eps**4*i*cos(pi*eps*(i - 1/2)) - 309237645312.0*pi**4*eps**4*i + 103079215104.0*pi**4*eps**4*cos(pi*eps*(i - 1/2)) + 38654705664.0*pi**4*eps**4 - 4123168604160.0*pi**3*eps**3*i**3*sin(pi*eps*(i - 1/2)) + 6184752906240.0*pi**3*eps**3*i**2*sin(pi*eps*(i - 1/2)) - 3092376453120.0*pi**3*eps**3*i*sin(pi*eps*(i - 1/2)) + 515396075520.0*pi**3*eps**3*sin(pi*eps*(i - 1/2)) - 2473901162496.0*pi**2*eps**2*i**2*cos(pi*eps*(i - 1/2)) + 4947802324992.0*pi**2*eps**2*i**2 + 2473901162496.0*pi**2*eps**2*i*cos(pi*eps*(i - 1/2)) - 4947802324992.0*pi**2*eps**2*i - 618475290624.0*pi**2*eps**2*cos(pi*eps*(i - 1/2)) + 1236950581248.0*pi**2*eps**2 - 4947802324992.0*pi*eps*i*sin(pi*eps*(i - 1/2)) + 2473901162496.0*pi*eps*sin(pi*eps*(i - 1/2)) - 4947802324992.0*cos(pi*eps*(i - 1/2)) + 4947802324992.0)/(pi**8*eps**6*(1099511627776.0*i**8 - 4398046511104.0*i**7 + 7696581394432.0*i**6 - 7696581394432.0*i**5 + 4810363371520.0*i**4 - 1924145348608.0*i**3 + 481036337152.0*i**2 - 68719476736.0*i + 4294967296.0))
elif C == 2:
    f1 = 6/eps**5*x**5 - 15/eps**4*x**4 + 10/eps**3*x**3  # x in [0, eps/2]
    f2 = 1  # x in [eps/2, 1-eps/2]
    f3 = 6/eps**5*(1-x)**5 - 15/eps**4*(1-x)**4 + 10/eps**3*(1-x)**3  # x in [1-eps/2, 1]
    
    
b1, b2 = eps/2, 1-eps/2
g1 = sp.sqrt(2)*sp.sin((2*i-1)*sp.pi*x)
g2 = sp.sqrt(2)*sp.cos((2*i-1)*sp.pi*x)
res = 0
for s, e, f in zip((0.0, b1, b2), (b1, b2, 1.0), (f1, f2, f3)):
    p = sp.integrate(f*g1, (x, s, e))**2 + sp.integrate(f*g2, (x, s, e))**2
    res += sp.simplify(p)
# print(res)
res = sp.simplify(res)
print(res)
res_simple = sp.simplify(res.subs(eps, 0.06632).evalf())
print()
print(res_simple)

