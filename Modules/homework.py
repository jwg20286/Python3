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
#Homework4
#INTEGRATION: HOW BIG ARE OBJECTS IN THE UNIVERSE?
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

#=======================================================
# Homework5
import numpy as np
import random
#=======================================================
# Monte carlo on birthday paradox
#=======================================================
# 1. Birthday paradox
result=dict() # store data in a dictionary
for N in range(40): # N is number of peoples, hear I've chosen it to be 0-39
    runN=10000 # number of samples to be averaged in each scenario
    matchCount=0

    for i in range(runN):
        birthdays=[]
        for i in range(N):
            birthdays.append(random.randint(1,365))
        if len(set(birthdays))!=N: #check if birthdays repeat
            matchCount+=1
    matchRate=matchCount/runN
    result.update({N:matchRate})
    resultSelect=dict((k,v) for k,v in result.items() if v>=0.5) # remove keys whose values are <0.5
minN=min(resultSelect,key=resultSelect.get) # find the key with the minimum value of all the values >=0.5
print(minN)
#===========================================================
# 2. calculate pi
#1/4 of a circle of radius 1 in 1x1 square, the square spans from (0,0) to (1,1) with side length 1.
N=10000000
insideCount=0 #how many points are inside the circle
for i in range(N):
    x=random.random()
    y=random.random()
    if x**2+y**2<=1: # same as np.sqrt(x**2+y**2)<=1
        insideCount+=1
Pi=insideCount/N*4
print(Pi)
#============================================================

#============================================================
# Homework6
#============================================================
import numpy as np
import random
#============================================================
# constants
fmp = 4*10**-3
c = 3.00*10**8

# measuring distance from the origin.
def distance(sr):
    """This function calculates the distance from the origin."""
    r = 0
    x = 0
    y = 0
    iterations = 0
    while r < sr:
        theta = np.random.uniform(0,2*np.pi)
        x += fmp*np.cos(theta)
        y += fmp*np.sin(theta)
        r = np.sqrt(x**2 + y**2)
        iterations += 1
    return (iterations*fmp)/c
#------------------------------------------------------------
#define radius, and number of trials to be averaged
sr=7
testN=20

totalTime=0
for i in range(testN):
    totalTime+=distance(sr)
aveTime=totalTime/testN # total amount to time to escape the sun
print('t=%.3e'%aveTime,' for sr=%f'%sr)
#------------------------------------------------------------
# The results I got are:
# R=0.7: average t=4.095e-07 
# R=1.4: average t=1.772e-06
# R=2.1: average t=3.675e-06
# R=3: average t=8.189e-06
# R=7: average t=3.883e-05
#------------------------------------------------------------
#data:
sr=np.array([0.7, 1.4, 2.1, 3, 7])
t=np.array([4.095e-07,1.772e-06,3.675e-06,8.189e-06, 3.883e-05])
# linear fit t vs sr^2
p=np.polyfit(sr**2,t,1)
slope=p[0]
intercept=p[1]
t_sun=p[0]*(7e8)**2+p[1]
t_sun_year=t_sun/(3600*24*365)
print('requires in total %f years'%t_sun_year) #total amount of years required to travel from core to surface

#plot data vs fit
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(sr**2,t,'.')
ax.plot(np.array([0,49]),np.array([intercept,slope*49+intercept]))
ax.set_xlabel('$r^2 (m^2)$')
ax.set_ylabel('Time (s)')
ax.grid()
plt.show()
#=============================================================

#=============================================================
#MIDTERM PROJECT
#DECODE
#=============================================================
# counts probability of letter c1 following letter c2, stored in p['c1']['c2']
# sum of 'c2' over p['c1']['c2'] will give 1, in other words, all values in p['c1'] add to 1.
#=====================================
#clean Text to remove special characters, lower all uppercase letters, replace all whitespaces with underscores
def cleanText(sampleText, trashText='''0123456789`~!@#$%^&*()-_=+[{]};:'",<.>/?'''): #trashText is optional, default is given
    sampleText=''.join([i for i in sampleText if i not in trashText]) #remove all characters belong to the trashText
    sampleText=sampleText.lower() #make all lowercase
    sampleText=sampleText.replace(' ','_') #replace whitespace with underscore
    return sampleText
#=====================================
# create alphabet ordered dictionary with all values 0
def dictAlphabet():
    alphabet='_'+string.ascii_lowercase
    dictAZ=od()
    for letter in alphabet:
        dictAZ[letter]=0
    return dictAZ
#=====================================
# create 2-layer alphabet dictionary with each value a 1-layer dictAlphabet style dictionary.
def dictAlphabet_2layer():
    alphabet='_'+string.ascii_lowercase
    p=od()
    for letter in alphabet:
        p[letter]=dictAlphabet()
    return p
#=====================================
#function to normalize a 1-layer dictionary
def dict_norm(dictionary):
    norm=np.array(list(dictionary.values())).sum() #normalization factor
    if norm==0:#if norm is 0
        norm=1 #make norm a non-zero value to avoid divide by 0 problem.
    for key in dictionary.keys():
        dictionary[key]/=norm
    return dictionary
#=======================================
# count how many times a letter l2 follow a letter l1, store in a 2-layer dictionary (1st index is l1, 2nd is l2)
# then normalize the counts to give probabilities
def followProb(sampleText,trashText='''0123456789`~!@#$%^&*()-_=+[{]};:'",<.>/?'''):
    sampleText=cleanText(sampleText,trashText=trashText)
    # create storage room as a 2 layer dictionary
    p=dictAlphabet_2layer()
    #------------------------------------------------
    # count instanced of l2 following l1
    for i in range(len(sampleText)-1):
        l1=sampleText[i] # 1st letter
        l2=sampleText[i+1] # 2nd letter
        p[l1][l2]+=1
    #normalize the dictionary for each l1
    for key in p.keys():
        p[key]=dict_norm(p[key])
    return p
#==========================================
#plot result
def probPlot(p):
    alphabet='_'+string.ascii_lowercase
    data=np.zeros([27,27,])
    for index in range(27):
        letter=alphabet[index]
        data[index]=np.array(list(p[letter].values())) #prepare 2d array for pcolor

    x=np.array([i for i in range(28)])
    y=np.array([i for i in range(28)])

    fig,ax=plt.subplots(1,1)
    c=ax.pcolor(x,y,data,cmap='BuGn',vmin=0,vmax=1)
    ax.set_title('conditional probability')
    ax.axis([0,27,0,27])

    ax.set_xlabel('second character')
    ax.set_ylabel('first character')
    ax.set_xticks(x+0.5)
    ax.set_yticks(y+0.5)
    ax.set_xticklabels(alphabet)
    ax.set_yticklabels(alphabet)

    fig.colorbar(c,ax=ax)
    plt.show()
#====================================================

