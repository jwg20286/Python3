#=======================================
#Homework1
#ELECTRIC POTENTIAL
import numpy as np
import matplotlib.pyplot as plt
#=======================================
# Calculate electric potential of a charge 'q', at a distance 'r'
def epotent(q,r):
    eps0=8.854187817e-12
    return 1/(4*np.pi*eps0)*q/r

# Calculate the combined e-potential at r1 from q1, and r2 from q2
def epotent2(q1,q2,r1,r2):
    return epotent(q1,r1)+epotent(q2,r2)

# Calculate the e-potential at (x,y), created by charge q1 at (x1,y1), and q2 at (x2,y2)
def epotent2xy(x,y,q1,x1,y1,q2,x2,y2):
    r1=np.sqrt((x-x1)**2+(y-y1)**2)
    r2=np.sqrt((x-x2)**2+(y-y2)**2)
    return epotent2(q1,q2,r1,r2)

#==========================================
# setup conditions
q1=-1
x1=-0.05
y1=0

q2=1
x2=0.05
y2=0

# create grid 
x=np.arange(-.5,+0.51,0.01)
y=np.arange(-.5,+0.51,0.01)

potential=np.zeros((101,101)) #data will be stored here

for indx in np.arange(101):
    for indy in np.arange(101):
        gridx=x[indx]
        gridy=y[indy]
        if not((indx==45)and(indy==50)): #skip left charge
            if not((indx==55)and(indy==50)): #skip right charge
                potential[indx][indy]=epotent2xy(gridx,gridy,q1,x1,y1,q2,x2,y2)
#==============================================
# plot using pcolor
fig,ax=plt.subplots(1,1)
c=ax.pcolor(x, y, potential.transpose(), cmap='RdBu', vmin=potential.min(), vmax=-potential.min())
ax.set_title('Electric potential')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(c, ax=ax)
#================================================
# ELECTRIC FIELD
#================================================
# calculate e-potential with fixed q1,x1,y1,q2,x2,y2 for this homework
def epxy(x,y):
    q1=-1
    x1=-0.05
    y1=0
    q2=1
    x2=0.05
    y2=0
    return epotent2xy(x,y,q1,x1,y1,q2,x2,y2)

# calculate h from r
def crt_h(r):
    #create h from r to fit the precision requirement
    return 2*np.cbrt(1e-16/6)*r

# calculate field by potential derivative
def ef(func_ep,x,y,h):
    #assume func_ep requires 2 inputs: x and y
    efx=(func_ep(x-h,y)-func_ep(x+h,y))/(2*h)
    efy=(func_ep(x,y-h)-func_ep(x,y+h))/(2*h)
    return efx,efy
#=================================================
# setup conditions
q1=-1
x1=-0.05
y1=0
q2=1
x2=0.05
y2=0

# create grid
x = np.arange(-0.5,+0.51,0.01)
y = np.arange(-0.5,+0.51,0.01)

# prepare storage matrices
fieldx=np.zeros((101,101)) #data will be stored here
fieldy=np.zeros((101,101))

for indx in np.arange(101):
    for indy in np.arange(101):
        gridx=x[indx]
        gridy=y[indy]
        if not((indx==45)and(indy==50)):
            if not((indx==55)and(indy==50)):
                rmax=np.sqrt(np.array([(gridx-x1)**2+(gridy-y1)**2,(gridx-x2)**2+(gridy-y2)**2]).max()) # use the farther charge to determine r
                h=crt_h(rmax)
                efx,efy=ef(epxy,gridx,gridy,h) # get x-&y-field components
                fieldx[indx][indy]=efx
                fieldy[indx][indy]=efy
                
fieldr=np.sqrt(fieldx**2+fieldy**2) # field r-component
fieldtheta=np.arctan2(fieldy,fieldx) # field direction from -pi to +pi
#====================================================
fig,ax=plt.subplots(1,2)

#plot amplitude
c0=ax[0].pcolor(x, y, fieldr.transpose(), cmap='RdBu', vmin=0, vmax=fieldr.max())
ax[0].set_title('Electric field amplitude')
ax[0].axis([x.min(), x.max(), y.min(), y.max()])
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
fig.colorbar(c0, ax=ax[0])

#plot direction
c1=ax[1].pcolor(x, y, fieldtheta.transpose(), cmap='RdBu', vmin=-np.pi, vmax=np.pi)
ax[1].set_title('Electric field amplitude')
ax[1].axis([x.min(), x.max(), y.min(), y.max()])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
fig.colorbar(c1, ax=ax[1])
#=====================================================

#=====================================================
#Homework1
#ELECTRIC POTENTIAL
import numpy as np
from numpy import ones,copy,cos,tan,pi,linspace
import warnings
import matplotlib.pyplot as plt
#=====================================================
#define H(z)
def h(z):
    omega_m=0.3
    omega_delta=0.7
    h0=2.2653721682847896e-18 #70000/3.09e22=this
    return h0*np.sqrt(omega_m*(1+z)**3+omega_delta)
#------------------------------------------------------
H=np.vectorize(h) #allow z to be sequences. The output will also be a sequence
#------------------------------------------------------
def Fch(z): #function c/H(z) in the integration
    c=299792458
    return c/H(z)
#=====================================================
# gauss quadrature finding x and w
def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w
#=====================================================
# integration functions
def integT(a,b,N,f):
    '''
    Trapezoidal integration.
    Syntax:
    -------
    integral=integT(a,b,N,f)
    
    Parameters:
    -----------
    a, b: float, lower, upper limit of integration
    N: int, number of bins
    f: func, function to be integrated.
    
    Returns:
    --------
    integral: float, integration result.
    '''
    h=(b-a)/N #size of each bin
    zlist=np.arange(a,a+h*(N+1),h) #create a list of z from a to b, equal spacing, N+1 points, N bins.
    flist=f(zlist) # create corresponding H(z) values, length=N+1
    return (flist[0]/2+flist[-1]/2+flist[1:-1:].sum())*h
#------------------------------------------------------
def integSR(a,b,N,f):
    '''
    Simpson's rule integration
    '''
    if N%2!=0:
        warnings.warn('\n N must be even.') # raise warning if N is not even
        
    h=(b-a)/N
    zlist=np.arange(a,a+h*(N+1),h) #create a list of z from a to b, equal spacing, N+1 points, N bins.
    flist=f(zlist) # create corresponding H(z) values, length=N+1
    return (flist[0]+flist[1:-1:2].sum()*4+flist[2:-1:2].sum()*2+flist[-1])*h/3
#------------------------------------------------------
def integG(a,b,N,f):
    '''
    Gauss quadrature integration
    '''
    x,w=gaussxwab(N,a,b)
    s=0
    for index in range(N):
        s+=w[index]*f(x[index])
    return s
#------------------------------------------------------
def integTrapz(a,b,N,f):
    '''
    np.trapz function integration
    '''
    h=(b-a)/N
    zlist=np.arange(a,a+h*(N+1),h) #create a list of z from a to b, equal spacing, N+1 points, N bins.
    flist=f(zlist) # create corresponding H(z) values, length=N+1
    return np.trapz(flist,x=zlist)
#==================================================
# determine bin numbers
NT=100
NSR=100
NG=100
NTrapz=100

# prepare storage space
yT=[]
ySR=[]
yG=[]
yTrapz=[]

# create DA(z) values
for z in np.arange(1,11): # z from 1 to 10
    dazT=integT(0,z,NT,H)/(1+z)
    dazSR=integSR(0,z,NSR,H)/(1+z)
    dazG=integG(0,z,NG,H)/(1+z)
    dazTrapz=integTrapz(0,z,NTrapz,H)/(1+z)
    yT+=[dazT]
    ySR+=[dazSR]
    yG+=[dazG]
    yTrapz+=[dazTrapz]
#=======================================================
# plot

z=np.arange(1,11)

fig,ax=plt.subplots(2,2)
fig.subplots_adjust(wspace=0.5,hspace=0.5)

ax[0][0].set_title('Trapezoid')
ax[0][0].plot(z,yT,'.')
ax[0][0].set_xlabel('z')
ax[0][0].set_ylabel('$D_A(z)$')
ax[0][0].grid()

ax[0][1].set_title('Simpson rule')
ax[0][1].plot(z,ySR,'.')
ax[0][1].set_xlabel('z')
ax[0][1].set_ylabel('$D_A(z)$')
ax[0][1].grid()

ax[1][0].set_title('Gauss quad')
ax[1][0].plot(z,yG,'.')
ax[1][0].set_xlabel('z')
ax[1][0].set_ylabel('$D_A(z)$')
ax[1][0].grid()

ax[1][1].set_title('numpy.trapz')
ax[1][1].plot(z,yTrapz,'.')
ax[1][1].set_xlabel('z')
ax[1][1].set_ylabel('$D_A(z)$')
ax[1][1].grid()
plt.show()
#=======================================================

