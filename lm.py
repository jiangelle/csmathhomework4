# -*- coding: utf-8 -*-
"""
Created on Sat May 07 10:13:17 2016

@author: jurcol
"""

"the Levenberg-Marquardt method to compute the minimum value of the function f(x)=2*x*x-4*x+5 "
#init the variables
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

x1= 0.1
x2 = 0.2
u = 1.

def func(x1,x2):
    return 5*pow(x1,2)+3*x1*x2+2*pow(x2,2)
    
def g(x1,x2):
    return np.array([10*x1+3*x2,3*x1+4*x2])
    
def G(x1,x2):
    return np.array([[10,3],[3,4]])

def q(x1,x2,s):
    return func(x1,x2)+np.dot(g(x1,x2),s)+1./2*np.dot(np.dot(s.T,G(x1,x2)),s)
    
while (la.norm(g(x1,x2),2) >= 1e-4):
    print 'func:' + str(func(x1,x2))
    while True:
        c = G(x1,x2)+u*np.eye(2)
        print c
        try :
            la.cholesky(c)
            break
        except:
            u = 4*u
    s = np.dot(la.inv(c),-g(x1,x2).T)    
    df = func(x1+s[0],x2+s[1])-func(x1,x2)
    dq = q(x1+s[0],x2+s[1],s) - q(x1,x2,np.array([0,0]).T)
    r = df/dq
    if r < 0.25:
        u = 4*u
    elif r > 0.75:
        u = u/2.
    if r > 0:
        x1 = x1 + s[0]
        x2 = x2 +s[1]
print x1,x2 ,func(x1,x2)

x,y=np.mgrid[-2:2:20j,-2:2:20j]
z=func(x,y)
#print x,y,z
ax=plt.subplot(111,projection='3d')
ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()