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
def cleanText(sampleText,keepspace=True):
    sampleText=''.join([i for i in sampleText if (i.isspace() or i in string.ascii_letters)]) #only keep whitespace and letters
    sampleText=sampleText.lower() #make all lowercase
    sampleText=sampleText.replace('\n',' ').replace('\r',' ') #replace line breakers with whitespace
    sampleText='_'.join(sampleText.split())
    if keepspace:
        return sampleText
    else:
        return sampleText.replace('_','')
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
def followProb(sampleText,keepspace=True):
    sampleText=cleanText(sampleText,keepspace=keepspace)
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
    #plt.show()
    return fig,ax
#====================================================
# energy of two characters, l1 followed by l2
def estr(l1,l2,p):
    if p[l1][l2]==0:
        return np.inf
    else:
        return -np.log(p[l1][l2]) #energy will be infinite if p[l1][l2]=0
#===============================
# energy of a string
def estr_long(string,p):
    prob=1
    for i in range(0,len(string)-1):
        l1=string[i]
        l2=string[i+1]
        prob*=p[l1][l2]
    if prob==0:
        return np.inf
    else:
        return -np.log(prob)
#=================================
# probability of a string happening
def pstr_long(string,p):
    prob=1
    for i in range(0,len(string)-1):
        l1=string[i]
        l2=string[i+1]
        prob*=p[l1][l2]
    return prob
#=================================
#accept something with a probability
def rolldice(prob):
    dice=random.random()
    if dice<prob:
        return True #accept
    else:
        return False #reject
#====================================
#probability of transfer from old to new state with different probability of happening
def tRate(pold,pnew,T):
    if pold>pnew:
        return (pnew/pold)**(1/T)
    else:
        return 1
#====================================
#probability of transfer from old to new state with different energy
def tRateE(eold,enew,T):
    pold=np.exp(-eold)
    pnew=np.exp(-enew)
    return tRate(pold,pnew,T)
#====================================
def generate_key(list_of_letters):
    # dictionary mapping random letter to true letter
    x=list_of_letters
    random.shuffle(x) # now the orders are different
    if '_' in x:
        key=dict(zip(x,'_'+string.ascii_lowercase))
        reverse_key=dict(zip('_'+string.ascii_lowercase, x))
    else:
        key=dict(zip(x,string.ascii_lowercase))
        reverse_key=dict(zip(string.ascii_lowercase, x))
    return key, reverse_key
#====================================
#generate key for our project
def autokey(keepspace=True):
    if keepspace:
        alphabet='_'+string.ascii_lowercase #assume using underscore and lowercase letters
    else:
        alphabet=string.ascii_lowercase #assume lowercase letters only
    lol=[l for l in alphabet]
    key,reverse_key=generate_key(lol)
    return key,reverse_key
#====================================
def generate_random_from_phrase(phrase, reverse_key): #use (phrase,reverse_key) to encrypt, and (jumbled_phrase,key) to decrypt
    jumbled_phrase = ''
    for i in range(len(phrase)):
        jumbled_phrase += reverse_key[phrase[i]]
    return jumbled_phrase
#=====================================
def repAlphabet(phrase,keepspace=True):
    counters=od(collections.Counter(phrase).most_common())
    if keepspace:
        alphabet='_'+string.ascii_lowercase
    else:
        alphabet=string.ascii_lowercase
    newAlphabet=''
    for l in alphabet:
        if l in counters:
            newAlphabet+=l*counters[l]
        else:
            newAlphabet+=l
    return newAlphabet
#=====================================
def compare(phrase1,phrase2):
    count=0
    for x, y in zip(phrase1,phrase2):
        if x==y:
            count+=1
    return count
#=============================================
def similarity(phrase1,phrase2):
    return compare(phrase1,phrase2)/len(phrase1)
#================================================
# randomly swap values of two keys in a given key-dictionary
def swap(key,alphabet='_'+string.ascii_lowercase):
    newkey={k:key[k] for k in key.keys()} #allocate memory for newkey
    length=len(alphabet)
    i1=random.randint(0,length-1)
    i2=random.randint(0,length-1)
    while alphabet[i2]==alphabet[i1]:
        i2=random.randint(0,length-1) #make sure l2!=l1
    l1=alphabet[i1]
    l2=alphabet[i2]
    newkey[l1],newkey[l2]=newkey[l2],newkey[l1]
    return newkey
#======================================
# update key if new key gives lower string energy
def updatekey(jumbledphrase,p,T,key,alphabet=string.ascii_lowercase):
    phrase_old=generate_random_from_phrase(jumbledphrase,key)
    newkey=swap(key,alphabet=alphabet)
    phrase_new=generate_random_from_phrase(jumbledphrase,newkey)
    #eold=estr_long(phrase_old,p)
    #enew=estr_long(phrase_new,p)
    #updateProb=tRateE(eold,enew,T)
    pold=pstr_long(phrase_old,p)
    pnew=pstr_long(phrase_new,p)
    updateProb=tRate(pold,pnew,T)
    #print(updateProb)
    if rolldice(updateProb):
        return newkey
    else:
        return key
#=========================================

#=========================================
#prepare hitchhiker text
f=open('/home/jwg20286/Downloads/hitchhiker.txt','r')
rawtext = f.read()
# calculate conditional probability
# keepspace
phht=followProb(rawtext,keepspace=True)
figt,axt=probPlot(phht)
axt.set_title(''' Hitchhiker's guide to the galaxy (with space)''')

phhf=followProb(rawtext,keepspace=False)
figf,axf=probPlot(phhf)
axf.set_title(''' Hitchhiker's guide to the galaxy (no space)''')

plt.show()
#=========================================
#prepare war and peace text
f=open('/home/jwg20286/Downloads/wp.txt','r')
rawtext = f.read()
# calculate conditional probability
# keepspace
pwpt=followProb(rawtext,keepspace=True)
figt,axt=probPlot(pwpt)
axt.set_title('War and peace (with space)')
# discard space
pwpf=followProb(rawtext,keepspace=False)
figf,axf=probPlot(pwpf)
axf.set_title('War and peace (no space)')
plt.show()
#============================================
#generate key and rkey to encrypt 'the_answer_to_life_the_universe_and_everything_is_forty_two'
key,rkey=autokey(keepspace=True)
p=pwpt

phrase='the_answer_to_life_the_universe_and_everything_is_forty_two'
jumbledphrase=generate_random_from_phrase(phrase,rkey)
op=generate_random_from_phrase(jumbledphrase,key) #original phrase, op==phrase
print(op)
print(jumbledphrase)
print(key)
# the key I generated was {'v': 'l', 'j': 's', 'w': 'r', 'r': 'c', 'b': 'p', 'o': 'n', 'a': 'u', 'k': 'm', 'm': 'b', 'e': 'w', 'u': 'k', 's': 'v', 'c': 'j', 'q': 'i', 'd': 'o', 't': 'y', 'g': 'e', 'i': 'g', 'p': 'd', 'l': 'h', '_': 't', 'y': 'z', 'f': 'a', 'n': 'q', 'h': 'x', 'x': 'f', 'z': '_'}.
#=============================================
#setting parameters for decryption
keepspace=True
# 1st process: weighted disturbance Markov chain
Tmax1=100
Tmin1=1
N1=10000000
tolerance1=0.95
# 2nd process: equal representation Markov chain
Tmax2=0.1
Tmin2=0.001
N2=0
tolerance2=0.95
#------------------------------------------------
#phrase='theanswertolifetheuniverseandeverythingisfortytwo'
#jumbledphrase='dhauxctaldikpwadhafxpgalcauxvagaledhpxrpcwildedti' #this corresponds to 'theanswertolifetheuniverseandeverythingisfortytwo'

phrase='the_answer_to_life_the_universe_and_everything_is_forty_two'
jumbledphrase='jodvsntid_vjrvxphdvjodvwnped_tdvsnzvded_yjopnqvptvhr_jyvjir' #the_answer_to_life_the_universe_and_everything_is_forty_two
#-------------------------------------------
if keepspace: # keep space
    p=pwpt
else: # no space
    p=pwpf
    
key,rkey=autokey(keepspace=keepspace) #no space
print('The original key is: ',key)

newAlphabet=repAlphabet(jumbledphrase,keepspace=keepspace) # generate repeated alphabet, more frequent letters in jumbledphrase are represented by more copies
print('The weighted alphabet is: ',newAlphabet)

n=0
while n<N1 and similarity(phrase,generate_random_from_phrase(jumbledphrase,key))<tolerance1:
    n+=1
    T=Tmax1**(1-n/N1)*Tmin1**(n/N1)
    key=updatekey(jumbledphrase,p,T,key,alphabet=newAlphabet) #non equal representation
print('The first process took %i steps.'%n)

n=0
while n<N2 and similarity(phrase,generate_random_from_phrase(jumbledphrase,key))<tolerance2:
    n+=1
    T=Tmax2**(1-n/N2)*Tmin2**(n/N2)
    key=updatekey(jumbledphrase,p,T,key) #equal representation
print('The second process took %i steps.'%n)
print('The deciphered key is: ',key)
#------------------------------------------
newphrase=generate_random_from_phrase(jumbledphrase,key)
print('\nThe original phrase is: ',phrase)
print('The jumbled phrase is: ',jumbledphrase)
print('The deciphered phrase is: ',newphrase)
print('The similarity between the two is: ', similarity(phrase,newphrase))
#=============================================
# notes on what I have done, and the results
#---------------------------------------------
# test 1, war and peace
'''
keepspace=True

Tmax1=100
Tmin1=1
N1=10000000
tolerance1=0.9

Tmax2=0.1
Tmin2=0.001
N2=100000
tolerance2=0.95
'''
#--------------
#The original key is:  {'w': 'q', 'v': 'k', 'j': 'i', 'a': 'b', 'r': 'a', 'b': 'u', 'k': 'r', 't': '_', 'o': 'j', 'm': 'x', 'e': 's', 's': 'p', 'q': 'd', 'd': 'g', 'g': 't', 'i': 'w', 'p': 'f', 'h': 'e', '_': 'n', 'y': 'l', 'c': 'o', 'n': 'y', 'x': 'c', 'l': 'z', 'u': 'm', 'z': 'h', 'f': 'v'}
#The weighted alphabet is:  ____abcddddddddeefghhiijjjjjjklmnnnnoooppppqrrrsstttuvvvvvvvvvvwxyyz
#The first process took 8812363 steps.
#The second process took 100000 steps.
#The deciphered key is:  {'v': '_', 'j': 't', 'w': 'u', 'r': 'o', 'a': 'z', 'b': 'f', 'k': 'p', 't': 's', 'o': 'h', 'm': 'm', 'e': 'v', 's': 'a', 'q': 'd', 'd': 'e', 'g': 'q', 'i': 'l', 'p': 'i', 'x': 'w', 'h': 'c', '_': 'r', 'y': 'y', 'c': 'k', 'n': 'n', 'l': 'x', 'u': 'j', 'z': 'g', 'f': 'b'}

#The original phrase is:  the_answer_to_life_the_universe_and_everything_is_forty_two
#The jumbled phrase is:  jodvsntid_vjrvxphdvjodvwnped_tdvsnzvded_yjopnqvptvhr_jyvjir
#The deciphered phrase is:  the_ansler_to_wice_the_universe_ang_everythind_is_corty_tlo
#The similarity between the two is:  0.8813559322033898
#===================
# test 2, war and peace
'''
keepspace=True

Tmax1=100
Tmin1=1
N1=10000000
tolerance1=0.9

Tmax2=0.1
Tmin2=0.001
N2=0
tolerance2=0.95
'''
#--------------------
#The original key is:  {'v': 'o', 'j': 'c', 'w': 'f', 'd': 'i', 'a': 'x', 'b': 'a', 'k': 'y', 's': 'd', 'o': 'j', 'm': 'l', 'e': 'p', 'u': 'n', 'c': 's', 'q': 'b', 'z': 'r', 't': 'w', 'g': 't', 'i': 'v', 'p': '_', 'h': 'm', '_': 'g', 'y': 'e', 'f': 'k', 'x': 'q', 'l': 'u', 'n': 'h', 'r': 'z'}
#The weighted alphabet is:  ____abcddddddddeefghhiijjjjjjklmnnnnoooppppqrrrsstttuvvvvvvvvvvwxyyz
#The first process took 8989805 steps.
#The second process took 0 steps.
#The deciphered key is:  {'v': '_', 'j': 't', 'w': 'w', 'd': 'e', 'b': 'b', 'k': 'f', 'a': 'u', 'o': 'h', 'm': 'k', 'e': 'v', 'u': 'p', 's': 'a', 'q': 'g', 'z': 'd', 't': 's', 'g': 'x', 'i': 'm', 'p': 'i', 'h': 'c', '_': 'r', 'y': 'y', 'c': 'z', 'n': 'n', 'l': 'q', 'x': 'l', 'r': 'o', 'f': 'j'}

#The original phrase is:  the_answer_to_life_the_universe_and_everything_is_forty_two
#The jumbled phrase is:  jodvsntid_vjrvxphdvjodvwnped_tdvsnzvded_yjopnqvptvhr_jyvjir
#The deciphered phrase is:  the_ansmer_to_lice_the_wniverse_and_everything_is_corty_tmo
#The similarity between the two is:  0.9152542372881356
#======================
# test 3, war and peace
'''
keepspace=True

Tmax1=100
Tmin1=1
N1=10000000
tolerance1=0.95

Tmax2=0.1
Tmin2=0.001
N2=0
tolerance2=0.95
'''
#-------------------------
#The original key is:  {'w': 'v', 'v': 'a', 'j': 'p', 's': 'n', 'r': 'f', 'a': 'k', 'b': 'b', 'k': 'y', 't': 'r', 'o': 'c', 'm': 'q', 'e': 'g', 'u': 'w', 'q': 'j', 'd': 's', 'g': 'z', 'i': 'h', 'p': '_', 'h': 'e', '_': 'u', 'y': 'x', 'c': 'd', 'x': 'i', 'l': 'm', 'n': 'l', 'z': 'o', 'f': 't'}
#The weighted alphabet is:  ____abcddddddddeefghhiijjjjjjklmnnnnoooppppqrrrsstttuvvvvvvvvvvwxyyz
#The first process took 10000000 steps.
#The second process took 0 steps.
#The deciphered key is:  {'v': '_', 'j': 't', 'w': 'b', 'r': 'i', 'b': 'j', 'o': 'h', 'a': 'q', 'k': 'x', 'm': 'w', 'e': 'r', 's': 'p', 'q': 'k', 'd': 'e', 't': 'd', 'g': 'o', 'i': 'v', 'p': 'a', 'x': 'c', 'h': 'g', '_': 'n', 'y': 's', 'c': 'f', 'n': 'l', 'l': 'm', 'u': 'u', 'z': 'y', 'f': 'z'}

#The original phrase is:  the_answer_to_life_the_universe_and_everything_is_forty_two
#The jumbled phrase is:  jodvsntid_vjrvxphdvjodvwnped_tdvsnzvded_yjopnqvptvhr_jyvjir
#The deciphered phrase is:  the_pldven_ti_cage_the_blarende_ply_erensthalk_ad_gints_tvi
#The similarity between the two is:  0.4576271186440678
#==========================
# test 4: war and peace
'''
keepspace=True

Tmax1=100
Tmin1=1
N1=10000000
tolerance1=0.95

Tmax2=0.1
Tmin2=0.001
N2=0
tolerance2=0.95
'''
#---------------------------------
#The original key is:  {'v': 'l', 'j': 's', 'w': 'r', 'r': 'c', 'b': 'p', 'o': 'n', 'a': 'u', 'k': 'm', 'm': 'b', 'e': 'w', 'u': 'k', 's': 'v', 'c': 'j', 'q': 'i', 'd': 'o', 't': 'y', 'g': 'e', 'i': 'g', 'p': 'd', 'l': 'h', '_': 't', 'y': 'z', 'f': 'a', 'n': 'q', 'h': 'x', 'x': 'f', 'z': '_'}
#The weighted alphabet is:  ____abcddddddddeefghhiijjjjjjklmnnnnoooppppqrrrsstttuvvvvvvvvvvwxyyz
#The first process took 10000000 steps.
#The second process took 0 steps.
#The deciphered key is:  {'v': '_', 'j': 't', 'w': 'u', 'r': 'o', 'a': 'j', 'b': 'z', 'k': 'w', 't': 'd', 'o': 'h', 'm': 'v', 'e': 'k', 's': 'i', 'q': 'g', 'd': 'e', 'g': 'm', 'i': 'l', 'p': 'a', 'l': 'x', '_': 'r', 'y': 'y', 'c': 'f', 'n': 'n', 'x': 'b', 'h': 'c', 'u': 'p', 'z': 's', 'f': 'q'}

#The original phrase is:  the_answer_to_life_the_universe_and_everything_is_forty_two
#The jumbled phrase is:  jodvsntid_vjrvxphdvjodvwnped_tdvsnzvded_yjopnqvptvhr_jyvjir
#The deciphered phrase is:  the_indler_to_bace_the_unakerde_ins_ekerythang_ad_corty_tlo
#The similarity between the two is:  0.711864406779661
#================================
# HOMEWORK 8
#================================
import numpy as np
import matplotlib.pyplot as plt
#================================
# gravitational acceleration due to M
def agravit(M,rm,rM): #rm, rM are 2d numpy arrays, returns acceleration by M on m
    G=6.67408e-11
    r=rM-rm
    R=np.sqrt((r**2).sum()) #length
    return G*M/R**3*r # 2d numpy array of F

# gravitational acceleration due to M1 and M2
def agravit2(M1,M2,rm,rM1,rM2):
    return agravit(M1,rm,rM1)+agravit(M2,rm,rM2)

# removed code
## create coordinate of an object orbiting the origin counter-clockwise starting from theta=0
## to get same thing but starting from theta=pi, use -d instead
#def xycc1(d,w,t):
#    return np.array([d*np.cos(w*t),d*np.sin(w*t)])
## create coordinates of an object orbiting the origin counter-clockwise starting from theta=0
#def xycc(d,w,t):
#    storage=np.array([[0.,0.],]*len(t))
#    for i in range(len(t)):
#        storage[i]=xycc1(d,w,t[i])
#    return storage
#-----------------------------------------
# runge-kutta 1 step, use rnow & tnow to output rnext & tnext(==tnow+h)
def rk1step(f,rnow,tnow,h):
    k1=h*f(rnow,tnow)
    k2=h*f(rnow+k1/2,tnow+h/2)
    k3=h*f(rnow+k2/2,tnow+h/2)
    k4=h*f(rnow+k3,tnow+h)
    rnext=rnow+(k1+2*k2+2*k3+k4)/6
    return tnow+h,rnext
#-----------------------------------------
# adaptive h
def updateh(f,rnow,tnow,h,error=1e-3):
    _,r1=rk1step(f,rnow,tnow,h)
    _,r2=rk1step(f,rnow,tnow,2*h)
    rho=30*h*error/np.sqrt(((r1-r2)**2).sum()) #displacement difference |x1-x2|
    return h*rho**(1/4)
#-----------------------------------------
# runge-kutta with optional adaptive time step
def rk2d(f,r0,t0,t1,h,error=1e-3,adaptive=False):
    t=np.array([t0,])
    r=np.array([r0,])
    while t[-1]<=t1:
        if adaptive:
            h=updateh(f,r[-1],t[-1],h,error=error)
        tnext,rnext=rk1step(f,r[-1],t[-1],h)
        t=np.append(t,tnext)
        r=np.append(r,[rnext],axis=0)
    return t,r
#================================
# actual problem
# three body problem
G=6.67408e-11
m_sun=1.989e30
m_jupiter=1.898e27
m1=0.68*m_sun
m2=0.2*m_sun
mp=0.33*m_jupiter
p12=41*24*3600
pp=229*24*3600 # make pp(period of the planet) smaller to move planet closer to the stars

u=m1+m2
d=((p12/np.pi/2)**2*G*u)**(1/3)
d1=d/0.88*0.2
d2=d/0.88*0.68
dp=((pp/np.pi/2)**2*G*u)**(1/3)
# print distance from the center of mass to the planet
print('dp=%s'%dp)

#-----------------------
#4d time derivative
def F(R,t): # R=(r1,r2,rp,v1,v2,vp)
    #r1=R[0:2]
    #r2=R[2:4]
    #r3=R[4:6]
    #v1=R[6:8]
    #v2=R[8:10]
    #v3=R[10:]
    # returns (v1,v2,vp,a1,a2,ap)
    return np.concatenate( ( R[6:],agravit2(m2,mp,R[0:2],R[2:4],R[4:6]),agravit2(m1,mp,R[2:4],R[0:2],R[4:6]),agravit2(m1,m2,R[4:6],R[0:2],R[6:8]) ) )
#-----------------------
#================================
# adjustable variables
t0=0 # initial time
t1=25*365.25*24*3600 # simulation stop time
h=48*3600 # initial time step
error=1e-3 # error level for adaptive time step
#----------------------------------------------
# initial positionss
r10=np.array([d1,0])
r20=np.array([-d2,0])
rp0=np.array([dp,0])
# initial velocity of the suns and the planet, assumed circular trajectory, constant speed, orbiting counter-clockwise
v10=np.array([0,2*np.pi*d1/p12])
v20=np.array([0,-2*np.pi*d2/p12])
vp0=np.array([0,2*np.pi*dp/pp]) # a=v^2/r => v=sqrt(a*r)
R0=np.concatenate( (r10,r20,rp0,v10,v20,vp0) ) # initial 12d vector
#-----------------------------------------------
t,R=rk2d(F,R0,t0,t1,h,error=error,adaptive=False)
#-----------------------------------------------
x1=np.array([])
y1=np.array([])
vx1=np.array([])
vy1=np.array([])

x2=np.array([])
y2=np.array([])
vx2=np.array([])
vy2=np.array([])

xp=np.array([])
yp=np.array([])
vxp=np.array([])
vyp=np.array([])
for i in range(len(R)):
    x1=np.append(x1,R[i][0])
    y1=np.append(y1,R[i][1])
    vx1=np.append(vx1,R[i][6])
    vy1=np.append(vy1,R[i][7])
    
    x2=np.append(x2,R[i][2])
    y2=np.append(y2,R[i][3])
    vx2=np.append(vx2,R[i][8])
    vy2=np.append(vy2,R[i][9])
    
    xp=np.append(xp,R[i][4])
    yp=np.append(yp,R[i][5])
    vxp=np.append(vxp,R[i][10])
    vyp=np.append(vyp,R[i][11])
#================================
# plot
fig,ax=plt.subplots(1,1)
ax.plot(x1[::],y1[::],'r.',markersize=1) #downsample to every several points
ax.plot(x2[::],y2[::],'b.',markersize=1)
ax.plot(xp[::],yp[::],'g.',markersize=1)
plt.show()
#================================

