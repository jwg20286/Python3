'''
Fitting function library.
All functions are designed to accept input fromat (f,*p), where p contains all additional parameters.
Input format (f,p) turns out to be inconvenient for scipy.optimize.curve_fit usage.
'''
import numpy as np
from numpy import pi,radians,sin,cos
import scipy.optimize
from scipy import fftpack
import warnings

#=======================================================================
def lrtzX(f,A,d,f0):
	f=np.array(f,float)
	A=float(A)
	d=float(d)
	f0=float(f0)
	x=A*d*f/4/pi**2/((d*f)**2+(f0**2-f**2)**2)
	return x
#=======================================================================
def lrtzY(f,A,d,f0):
	f=np.array(f,float)
	A=float(A)
	d=float(d)
	f0=float(f0)
	y=A*(f0**2-f**2)/4/pi**2/((d*f)**2+(f0**2-f**2)**2)
	return y
#=======================================================================
def lrtzXph(f,A,d,f0,phase):
	f=np.array(f,float)
	A=float(A)
	d=float(d)
	f0=float(f0)
	phase=float(phase)
	x=lrtzX(f,A,d,f0)*cos(radians(phase))-lrtzY(f,A,d,f0)*sin(radians(phase))
	return x
#=======================================================================
def lrtzYph(f,A,d,f0,phase):
	f=np.array(f,float)
	A=float(A)
	d=float(d)
	f0=float(f0)
	phase=float(phase)
	y=lrtzX(f,A,d,f0)*sin(radians(phase))+lrtzY(f,A,d,f0)*cos(radians(phase))
	return y
#=======================================================================
def lrtzRR(f,A,d,f0):
	f=np.array(f,float)
	A=float(A)
	d=float(d)
	f0=float(f0)
	rr=(A/4/pi**2)**2/((d*f)**2+(f0**2-f**2)**2)
	return rr
#=======================================================================
def bgCon(f,a0):
	a0=float(a0)
	return a0
#=======================================================================
def bgLin(f,a1):
	f=np.array(f,float)
	a1=float(a1)
	return a1*f
#=======================================================================
def bgSq(f,a2):
	f=np.array(f,float)
	a2=float(a2)
	return a2*f*f
#=======================================================================
def bgCub(f,a3):
	f=np.array(f,float)
	a3=float(a3)
	return a3*f*f*f
#=======================================================================
def bgInv(f,c1):
	f=np.array(f,float)
	c1=float(c1)
	return c1/f
#=======================================================================
def bgInv2(f,c2):
	f=np.array(f,float)
	c2=float(c2)
	return c2/f/f
#=======================================================================
def bgInv3(f,c3):
	f=np.array(f,float)
	c3=float(c3)
	return c3/f/f/f
#=======================================================================
def PLTS2000T2P(T,Pn=34.3934):
	'''
	Temperature to pressure in PLTS-2000 scale. Limit is 0.9mK-1K.
	For program time efficiency, T limit validation is not checked.
	Syntax: 
	-------
	P=PLTS2000T2P(T[,Pn=34.3934])
	Parameters:
	-----------
	T: Temperature in mK, sequence aware.
	Pn: Neel transition pressure in bar. Default is 34.3934 bar.
	Returns:
	--------
	P: Pressure in bar.
	'''
	T=np.array(T,dtype=float)/1000 #convert T to K required by PLTS-2000
	c3=-1.3855442e-12
	c2=4.5557026e-9
	c1=-6.4430869e-6
	a0=3.4467434
	a1=-4.4176438
	a2=1.5417437e1
	a3=-3.5789853e1
	a4=7.1499125e1
	a5=-1.0414379e2
	a6=1.0518538e2
	a7=-6.9443767e1
	a8=2.6833087e1
	a9=-4.5875709

	P=c3/T**3+c2/T**2+c1/T+a0+a1*T+a2*T**2+a3*T**3+a4*T**4+a5*T**5+a6*T**6+a7*T**7+a8*T**8+a9*T**9
	P=P*10-34.3934+Pn #convert P from MPa(given by PLTS-2000) to bar, and shift it based on a measured Pn.
	return P
#=======================================================================
def PLTS2000P2T(P,Pn=34.3934):
	'''
	Pressure to temperature in PLTS-2000 scale. Limit is 0.680014mK-1K.
	Syntax:
	-------
	T=PLTS2000P2T(P[,Pn=34.3934])
	Parameters:
	-----------
	P: Pressure in bar, sequence aware.
	Pn: Neel transition pressure in bar.
	Returns:
	--------
	T: np.array of calculated temperatures. For scalar P input: T=[Tlow,Thigh], for sequence P input: T=[[Tlows],[Thighs]], even if len(P)==1.
	Note:
	-----
	P lower than model allowed minimum will be changed to the minimum.
	P higher than model allowed low T branch max will return Tlow=nan, this model allowed max P happened at T=0.680014mK.
	The valid range, [0.9,1000]mK, is smaller than the calculable range.
	'''
#-----------------------------------------------------------------------
# single run function that solves for T with input P using PLTS2000 scale with a given Neel transition pressure Pn.
	def PLTS2000P2T_single(P,Pn):
		Pmin=PLTS2000T2P(315.23959351,Pn=Pn) #minimal pressure in this model.
		P0=PLTS2000T2P(0.680014,Pn=Pn) #max pressure on low T branch, for a T lower than 0.680014mK, the model-generated P decreases with T.
		if P<Pmin: #check if P is no smaller than model allowed minimum
			warnings.warn('\nPressure must be higher than %.12f..bar.\nInput value will be auto-converted to %.12f..bar.'%(Pmin,Pmin))
			P=Pmin #make P as least Pmin
	
		f=lambda T: PLTS2000T2P(T,Pn=Pn)-P #build function to solve its roots
	
		# Low T branch solution:
		if P<=P0: #if P>P(T=0.690014mK,Pn=Pn), return nan
			Tlow=scipy.optimize.brentq(f,0.680014,315.23959351)
		else:
			Tlow=np.nan #return Tlow=nan because no root on low T branch
		# High T branch solution:
		Thigh=scipy.optimize.brentq(f,315.23959351,1000)

		return np.array([Tlow,Thigh])
#-----------------------------------------------------------------------
# wrap around PLTS2000P2T_single to accomodate np.array input and output
	if not(isinstance(P,(list,tuple,np.ndarray))): #input is scalar
		return PLTS2000P2T_single(P,Pn) #return as [Tlow,Thigh]
	
	P=np.array(P,dtype=float) #input is sequence, including sequences with length=1
	T=np.array([])
	for pp in P:
		tt=PLTS2000P2T_single(pp,Pn)
		T=np.append(T,tt)
	return np.array([T[0::2],T[1::2]]) #return as [[Tlows],[Thighs]]
#=======================================================================
def floridaT2P(T,Pn=34.3934):
	'''
	Florida temperature scale: T(bar)->P(bar). Only valid for T<Tn and T=Tn.
	Syntax:
	-------
	P=floridaT2P(T[,Pn=34.3934])
	Parameters:
	-----------
	T: temperature in mK, sequence aware.
	Pn: Neel transition pressure in bar.
	Returns:
	--------
	P: pressure in bar.
	Note:
	-----
	The florida scale is valid for T within [0.5,25]mK. With one formula above Tn and a different one below Tn. This program used a shifted version of the formula below Tn(T within [0.5,Tn]mK) to match the PLTS2000 T=0.9mK value. Here Tn=0.9061261416052492!=0.902mK due to the shift.
	'''
	T=np.array(T,dtype=float)
	PnkPa=Pn*100 #bar to kPa
	b0=0.17601979525638645 #b0=0.2611*0.9**4+(PLTS2000T2P(0.9,Pn=Pn)-Pn)*100, is shifted from the original Florida scale value to match with PLTS2000 tablular values. When T=0.9mK which is the lower bound of PLTS2000 scale, P_florida(T=0.9mK)=P_PLTS2000(T=0.9mK), so that the two scales are continuous at 0.9mK. This value is independent of the choice of Pn.
	b4=-0.2611
	PkPa=PnkPa+b0+b4*T**4
	P=PkPa/100 #return in bar
	return P
#=======================================================================
def floridaP2T(P,Pn=34.3934):
	'''
	Find T from P using florida scale. The used scale is the below Tn branch of florida scale. Solved T that's not in the range [0,0.902]mK is returned as np.nan.
	Syntax:
	-------
	T=floridaP2T(P[,Pn=34.3934])
	Parameters:
	-----------
	P: pressure in bar, sequence aware.
	Pn: Neel transition pressure in bar.
	Returns:
	--------
	T: temperature in mK.
	Note:
	-----
	The valid range for the formula is [0.5mK,Tn], here Tn=0.9061261416052492!=0.902mK due to the shift. The solutions within [0,0.5)mK are also returned for the sake of keeping as many number outputs as possible. The scale overlaps with PLTS2000 above 0.9mK, for which the PLTS2000 is preferred over florida scale.
	'''
#-----------------------------------------------------------------------
# single run function that solves for T with input P using florida scale with a given Neel transition pressure Pn.
	def floridaP2T_single(P,Pn):
		f=lambda T:floridaT2P(T,Pn=Pn)-P
		if f(0.9061261416052492)>0: #P below its Tn value
			T=np.nan
		elif f(0)<0: #P above its zero-T value.
			T=np.nan
		else: #P(T=0)>=P>=P(T=Tn)
			T=scipy.optimize.brentq(f,0,0.9061261416052492) #solve: f=0
		return T
#-----------------------------------------------------------------------
# wrap around floridaP2T_single to accomodate np.array input and output
	P=np.array(P,dtype=float)

	wrapFloridaP2T=np.vectorize(floridaP2T_single)
	T=wrapFloridaP2T(P,Pn)
	return T
#=======================================================================
def he3P2Nu(P,deg=5):
	'''
	Calculate molar volume(cm^3/mol) from given pressure(bar).
	Syntax:
	-------
	Nu,popt=he3P2Nu(P[,deg=5])
	Parameters:
	-----------
	P: Pressure in bar.
	deg: np.polyfit degree, polynomial order to fit the Halperin data.
	Returns:
	--------
	Nu: Molar volume in cm^3/mol.
	popt: Fitted polynomial parameters, lower order terms in front.
	Notes:
	------
	deg=5 because test fits show this to be sufficient. deg<5 will cause inaccuracy. deg>5 can be done, but may be unnecessary.
	'''
	x=np.linspace(0,34,35)
	y=[36.818, 35.715, 34.761, 33.934, 33.215,\
	32.587, 32.036, 31.547, 31.110, 30.716,\
	30.357, 30.025, 29.716, 29.426, 29.150,\
	28.888, 28.638, 28.398, 28.168, 27.949,\
	27.739, 27.541, 27.354, 27.177, 27.012,\
	26.857, 26.711, 26.572, 26.438, 26.306,\
	26.170, 26.025, 25.864, 25.677, 25.456]#cite Halperin1990
	popt=np.polyfit(x,y,deg)[::-1] #poly fit, lower order terms in front
	Nu=np.polynomial.polynomial.polyval(P,popt) #molar volume(cm^3/mol)
	return Nu, popt
#=======================================================================
def domainfft(t):
	'''
	Generate FFT f-domain based on t-domain's number of points and its sampling rate.
	Syntax:
	-------
	f=fftt2f(t)
	Parameters:
	-----------
	t: t-domain data array.
	Returns:
	--------
	f: f-domain data array.
	Note:
	-----
	This program works for both f2t and t2f. There is no difference in fft and ifft in terms of domain transition.
	'''
	numpts=len(t) #number of points
	tstep=(max(t)-min(t))/(numpts-1) # assuming constant step
	fmax=1/tstep
	fstep=fmax/numpts
	f=np.linspace(0,fstep*(numpts-1),numpts)
	return f
#=======================================================================
def FID(t,s0,T,f0,phase):
	'''
	Create a free-induction-decay spectrum from time with input parameters.
	Syntax:
	-------
	wave=FID(t,s0,T,f0,phase)
	Parameters:
	-----------
	t: Time array.
	s0: Amplitude of waveform.
	T: Decay time constant.
	f0: Frequency of FID's oscillation.
	phase: Initial phase offset (degree).
	Returns:
	--------
	wave: waveform array, calculated from time, len(wave)==len(t).
	'''
	p=np.deg2rad(phase)
	return s0*np.exp(-t/T)*np.cos(2*np.pi*f0*t+p)
#=======================================================================
def FID0(t,s0,T,f0,phase,zerofillnum=0):
	'''
	Create a free-induction-decay spectrum from time with input parameters, and append zeros to it.
	Syntax:
	-------
	wave0fill=FID0(t,s0,T,f0,phase[,zerofillnum=0])
	Parameters:
	-----------
	t: Time array.
	s0: Amplitude of waveform.
	T: Decay time constant.
	f0: Frequency of FID's oscillation.
	phase: Initial phase offset (degree).
	zerofillnum: number of extra zero points appended to FID.
	Returns:
	--------
	wave0fill: zero filled FID.
	'''
	wave=FID(t,s0,T,f0,phase)
	wave0fill=np.append(wave,np.zeros(zerofillnum)) #zerofilling wave
	return wave0fill
#=======================================================================
def FID0s(t,*p,zerofillnum=0):
	'''
	Create a superposed spectrum of multiple FIDs from time with the same zerofilling.
	Syntax:
	-------
	wave0fill=FID0s(t,*p[,zerofillnum=0])
	Parameters:
	-----------
	t: Time array.
	p: [t1,s01,T1,f01,t2,s02,T2,f02,t3,s03,T3,f03,...], 4xN long array, decribing each FID.
	zerofillnum: number of extra zero points appended to FID.
	Returns:
	--------
	wave0fill: superposed zero-filled FID spectrum.
	'''
	p=np.array(p)
	numpk=int(p.size/4) #number of peaks
	wave0fill=0
	for i in range(0,numpk):
		wave0fill+=FID0(t,*p[i*4:i*4+4:],zerofillnum=zerofillnum)
	return wave0fill
#=======================================================================
def FIDfft(f,s0,T,f0,phase):
	'''
	Create a fast fourier transform of free-induction-decay spectrum from frequency with input parameters.
	Syntax:
	-------
	wavefft=FIDfft(f,s0,T,f0,phase)
	Parameters:
	-----------
	f: Frequency array.
	s0: Amplitude of waveform.
	T: Decay time constant.
	f0: Frequency of FID's oscillation.
	phase: Initial phase offset (degree).
	Returns:
	--------
	wavefft: FFT of wave, exists in the frequency domain, len(wavefft)==len(f).
	'''
	t=domainfft(f)
	wave=FID(t,s0,T,f0,phase)
	return fftpack.fft(wave)
#=======================================================================
def FID0fft(f,s0,T,f0,phase,zerofillnum=0):
	'''
	Create a fast fourier tranform of a zero-filled free-induction-decay spectrum from frequency with input parameters.
	Syntax:
	-------
	wave0fillfft=FID0fft(f,s0,T,f0,phase[,zerofillnum=#)
	Parameters:
	-----------
	f: Frequency array, before zerofilling.
	s0: Amplitude of waveform.
	T: Decay time constant.
	f0: Frequency of FID's oscillation.
	phase: Initial phase offset (degree).
	zerofillnum: number of extra zero points appended to FID.
	Defaults: zerofillnum=0.
	Returns:
	--------
	wave0fillfft: FFT of zero-filled FID.
	'''
	t=domainfft(f)
	wave0fill=FID0(t,s0,T,f0,phase,zerofillnum=zerofillnum)
	wave0fillfft=fftpack.fft(wave0fill)
	return wave0fillfft
#=======================================================================
def FID0ffts(f,*p,zerofillnum=0):
	'''
	Create a FFT of a superposed spectrum of multiple FIDs with the same zerofilling.
	Syntax:
	-------
	wave0fillfft=FID0ffts(f,*p[,zerofillnum=0])
	Parameters:
	-----------
	f: Frequency array, before zerofilling.
	p: [t1,s01,T1,f01,t2,s02,T2,f02,t3,s03,T3,f03,...], 4xN long array, decribing each FID.
	zerofillnum: number of extra zero points appended to FID.
	Returns:
	--------
	wave0fillfft: FFT of superposed zero-filled FIDs.
	'''
	p=np.array(p)
	numpk=int(p.size/4)#number of peaks
	wave0fillfft=0
	for i in range(0,numpk):
		wave0fillfft+=FID0fft(f,*p[i*4:i*4+4:],zerofillnum=zerofillnum)
	return wave0fillfft
#=======================================================================
