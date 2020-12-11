'''
Utility use functions. Designed to NOT have dependency on any home-made modules.
'''
import os, time
import numpy as np
from numpy import isnan,exp,log
import pandas as pd
import FuncLib
from pint import UnitRegistry
ureg=UnitRegistry()
import scipy.special

#=======================================================================
def mctC2P(C,p):
	'''
	Calculate mct pressure, P(psia), from capacitance, C(pF), and calibration parameters, p.
	Syntax:
	-------
	P=mctC2P(C,p)
	Parameters:
	-----------
	C: mct capacitance measurement, unit is pF.
	p: prefactor parameters. P=p0+p1*C+p2*C^2+...+pn*C^n.
	Returns:
	--------
	P: mct pressure, unit is bar.
	'''
	Ppsi=np.polynomial.polynomial.polyval(C,p)
	
	f=lambda P:(P*ureg.psi).to(ureg.bar).magnitude #converts P from psi to bar, automatically sequence aware thanks to ureg. A direct division is more efficient here, but ureg is implemented in case future configurability is needed.
	Pbar=f(Ppsi)
	return Pbar
#=======================================================================
def mctP2T(P,branch='low',Pn=34.3934):
	'''
	Calculate mct temperature from its pressure.
	Syntax:
	-------
	T=mctP2T(P[,branch='low',Pn=34.3934])
	Parameters:
	-----------
	P: pressure in bar.
	branch: the melting curve branch in which the solution is searched for. 'low' is below 315.23959351mK, which is the lowest pressure point on the melting curve, 'high' is above this point.
	Pn: Neel transition pressure in bar.
	Returns:
	--------
	T: temperature in mK.
	'''
# if branch=='high':
	def mctP2Thigh(P,Pn):#sequence aware
		T=FuncLib.PLTS2000P2T(P,Pn=Pn)[1] #high branch PLTS2000 only
		return T
#-----------------------------------------------------------------------
# if branch=='low'
	def mctP2Tlow(P,Pn):#only takes scalar input
		if P<=FuncLib.PLTS2000T2P(0.9,Pn=Pn): #low P side of low branch
			T=FuncLib.PLTS2000P2T(P,Pn=Pn)[0]
		else: #high P side of low branch
			T=FuncLib.floridaP2T(P,Pn=Pn)
		return T
#-----------------------------------------------------------------------
	if branch=='high': #melting curve 'high' branch
		return mctP2Thigh(P,Pn)
	else: # 'low' branch
		f=np.vectorize(mctP2Tlow)
		return f(P,Pn)
#=======================================================================
def mctC2T(C,p,branch='low',Pn=34.3934):
	'''
	Calculate temperature, T, from capacitance, C.
	Syntax:
	-------
	T=mctC2T(C,p[,branch='low',Pn=34.3934])
	Parameters:
	-----------
	C: capacitance in pF.
	p: prefactor parameters for mctC2P. P=p0+p1*C+p2*C^2+...+pn*C^n. 
	branch: the melting curve branch in which the solution is searched for. 'low' is below 315.23959351mK, which is the lowest pressure point on the melting curve, 'high' is above this point.
	Pn: Neel transition pressure in bar.
	Returns:
	--------
	T: temperature in mK.
	Notes:
	------
	The returned T may contain np.nan values.
	'''
	P=mctC2P(C,p)
	T=mctP2T(P,branch=branch,Pn=Pn)
	return T
#=======================================================================
def chooseT(Tmc,Tmct):
	'''
	Syntax:
	-------
	Tmm=chooseT(Tmc,Tmct)
	Parameters:
	-----------
	Tmc,Tmct: mixing chamber temperature, MCT temperature, units are mK.
	Returns:
	--------
	Tmm:returns Tmc to Tmm when max(Tmc,Tmct)>100mK
		returns Tmct to Tmm when both below 100mK
		if one of them is NAN, returns the other to Tmm
		if both are NAN, raise an error
	'''
	if isnan(Tmc) and isnan(Tmct):
		raise TypeError('Both values are N/A values') #raise error if both values are NAN
	elif isnan(Tmc) or isnan(Tmct):
		Tmm=np.nanmax([Tmc,Tmct]) #if one is NAN, return the other to Tmm
	elif np.maximum(Tmc,Tmct)>100: #b/c in this region, Tmc<Tmct==>this condition is the same as Tmct>100
		Tmm=Tmc #b/c Tmct switches around 315mK, the delay in manually switch solutions may cause Tmc>Tmct
	else:
		Tmm=Tmct
	return Tmm
#=======================================================================
def colorCode(iter):
	'''
	return color codes in HEX string
	support max 24 colors
	check http://colorbrewer2.org/
	or http://tools.medialab.sciences-po.fr/iwanthue/
	for more details.
	'''
	cc=["#ff1c44","#55ff5f","#a300af","#fff44d","#4a61d7","#ffa70e","#f066ff","#009732","#ff3fbf","#54ffbb","#990077","#bda300","#a783ff","#004105","#ff9bf4","#02eff3","#823628","#d4ffee","#4b003f","#ffb5a4","#162a3f","#009fea","#351400","#002f6b"]
	return cc[iter]
#=======================================================================
def lencc():
	'''
	returns the length of colorCode iteration length len(cc).
	'''
	return 24
#=======================================================================
def gainCorrect(f,rawdata):
	'''
	2017-06-23 11:01
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=200mVrms
	1/0.16=2.5/(2*0.2)

	Syntax:
	-------
	newdata=gainCorrect(f,rawdata)
	Parameters:
	-----------
	f: frequency in Hz, will be converted to numpy.ndarray of dtype=float.
	rawdata: rawdata points array with the same length as f.
	Returns:
	--------
	newdata: rolloff gain corrected data.
	'''
	f=np.array(f,dtype=float)
	rawdata=np.array(rawdata,dtype=float)
	newdata=0.16*rawdata/gainVsF1(f)
	return newdata
#=======================================================================
def gainCorrect_5mVrms(f,rawdata):
	'''
	2020-10-23 14:56
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=5mVrms
	1/0.004=2.5/(2*0.005)

	Syntax:
	-------
	newdata=gainCorrect_5mVrms(f,rawdata)
	Parameters:
	-----------
	f: frequency in Hz, will be converted to numpy.ndarray of dtype=float.
	rawdata: rawdata points array with the same length as f.
	Returns:
	--------
	newdata: rolloff gain corrected data.
	'''
	f=np.array(f,dtype=float)
	rawdata=np.array(rawdata,dtype=float)
	newdata=0.004*rawdata/gainVsF1(f)
	return newdata
#=======================================================================
def gainCorrect_10mVrms(f,rawdata):
	'''
	2020-12-11 14:39
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=10mVrms
	1/0.008=2.5/(2*0.01)

	Syntax:
	-------
	newdata=gainCorrect_5mVrms(f,rawdata)
	Parameters:
	-----------
	f: frequency in Hz, will be converted to numpy.ndarray of dtype=float.
	rawdata: rawdata points array with the same length as f.
	Returns:
	--------
	newdata: rolloff gain corrected data.
	'''
	f=np.array(f,dtype=float)
	rawdata=np.array(rawdata,dtype=float)
	newdata=0.008*rawdata/gainVsF1(f)
	return newdata
#=======================================================================
def gainCorrect_20mVrms(f,rawdata):
	'''
	2017-06-23 11:09
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=20mVrms
	1/0.016=2.5/(2*0.02)

	Syntax:
	-------
	newdata=gainCorrect_20mVrms(f,rawdata)
	Parameters:
	-----------
	f: frequency in Hz, will be converted to numpy.ndarray of dtype=float.
	rawdata: rawdata points array with the same length as f.
	Returns:
	--------
	newdata: rolloff gain corrected data.
	'''
	f=np.array(f,dtype=float)
	rawdata=np.array(rawdata,dtype=float)
	newdata=0.016*rawdata/gainVsF1(f)
	return newdata
#=======================================================================
def gainCorrect_50mVrms(f,rawdata):
	'''
	2017-07-13 13:30
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=50mVrms
	1/0.04=2.5/(2*0.05)

	Syntax:
	-------
	newdata=gainCorrect_50mVrms(f,rawdata)
	Parameters:
	-----------
	f: frequency in Hz, will be converted to numpy.ndarray of dtype=float.
	rawdata: rawdata points array with the same length as f.
	Returns:
	--------
	newdata: rolloff gain corrected data.

	'''
	f=np.array(f,dtype=float)
	rawdata=np.array(rawdata,dtype=float)
	newdata=0.04*rawdata/gainVsF1(f)
	return newdata
#=======================================================================
def gainCorrect_500mVrms(f,rawdata):
	'''
	2017-07-13 13:30
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=500mVrms
	1/0.4=2.5/(2*0.5)

	Syntax:
	-------
	newdata=gainCorrect_50mVrms(f,rawdata)
	Parameters:
	-----------
	f: frequency in Hz, will be converted to numpy.ndarray of dtype=float.
	rawdata: rawdata points array with the same length as f.
	Returns:
	--------
	newdata: rolloff gain corrected data.

	'''
	f=np.array(f,dtype=float)
	rawdata=np.array(rawdata,dtype=float)
	newdata=0.4*rawdata/gainVsF1(f)
	return newdata
#=======================================================================
def gainVsF1(f):
	'''
	The frequency dependent demodulation gain of SR7124.
	This model is an exponential with an index of 6th-order polynomial.
	SR7124: sens=200Vrms, Tc=10us, slope=12dB/Oct.
			Carrier frequency (w1/2pi)=150kHz.
	Note: DAC1 of SR7124 outputs 2.5Vdc is the full scale sense.
	Syntax:
	-------
	gain=gainVsF1(f)
	Parameters:
	-----------
	f: the modulating frequency (w2/2pi) in Hz in 1-40kHz range,	f list/tuple/numpy.ndarray will be converted to numpy.ndarray of dtype=float
	Returns:
	--------
	gain: the ratio of output (Ro*cos(w2t)) to input (Ri*cos(w1t)cos(w2t).
	gain=Ro(mVrms)*sqrt(2)*200/(2500*Ri(mVrms)).
	'''
	f=np.array(f,dtype=float)
	a0=0.0113641
	a1=-3.13929e-6
	a2=-2.62802e-10
	a3=-9.26438e-15
	a4=-2.47202e-19
	a5=1.39815e-23
	a6=-2.29362e-28
	gain=exp(log(10)*(a0+a1*f+a2*f**2+a3*f**3+a4*f**4+a5*f**5+a6*f**6))
	return gain
#=======================================================================
def linestyleCode(iter):
	'''
	return linestyle codes in string
	support max 4 styles
	check http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D.set_linestyle for details
	'''
	ls=['-','--','-.',':']
	return ls[iter]
#=======================================================================
def markerCode(iter):
	'''
	return maker codes in string
	support max 23 styles
	check http://matplotlib.org/api/markers_api.html for details
	'''
	mc=['o','^','s','D','<','p','v','*','8','+','>','x','.','H','4',',','d','|','h','1','_','2','3']
	return mc[iter]
#=======================================================================
def mkAxLabel(labelstr):
	'''
	make labels for axis titles
	translate labelstr and return output to axlabel
	this function will translate:
		'f/F'=>'Frequency (Hz)'
		'x/y/r/X/Y/R'=>'X/Y/R-channel (Vpp)'
	Syntax:
	-------
	axlabel=mkAxLabel(labelstr)
	Parameters:
	-----------
	labelstr: case insensitive label string of 'f/x/y/r'.
	Returns:
	--------
	axlabel: axis label translated from input label.
	'''
	if labelstr in 'fF': #label for 'f'
		axlabel='Frequency (Hz)'
	elif labelstr in 'xyrXYR': #label for 'x/y/r'
		axlabel=labelstr.upper()+'-channel (Vpp)'
	else:
		raise TypeError('Unrecognizable input string')
	return axlabel
#=======================================================================
def mkNmrAxLabel(labelstr):
	'''
	'''
	str=labelstr.lower()
	if str=='t':
		axlabel='Time (s)'
	elif str=='d':
		axlabel='FID Magnitude (V)'
	elif str=='f':
		axlabel='Frequency (Hz)'
	elif str=='r':
		axlabel='FFT FID Real Component (a.u.)'
	elif str=='i':
		axlabel='FFT FID Imaginary Component (a.u.)'
	elif str=='m':
		axlabel='FFT FID Magnitude (a.u.)'
	elif str=='p':
		axlabel='FFT FID Phase (a.u.)'
	else:
		raise TypeError('Unrecognizable input string')
	return axlabel
#=======================================================================
def mkFilename(device,filenum):
	'''
	concatenate device code and filenum in a device_filenum.dat fashion
	Syntax:
	-------
	filename=mkFilename(device,filenum):
	Parameters:
	-----------
	device: device code string, e.g. 'h1m','TF1201'
	filenum: file number, filled to 3 digits if shorter than 3
	Returns:
	--------
	filename: file name.
	'''
	filename=device+'_%0.3d.dat'%filenum
	return filename
#=======================================================================
def sweepLoad(filename,ftimes=1,xtimes=1,ytimes=1,rtimes=1):
	'''
	Loads file with 'filename'.
	Syntax:
	-------
	f,x,y,r,Tmc,Tmct,Cmct=sweepLoad(filename[,ftimes=1,xtimes=1,ytimes=1,rtimes=1])
	Parameters:
	-----------
	filename: Filename to read from.
	f/x/y/rtimes: The raw data will be multiplied by this.
	Returns:
	--------
	f: Frequency.
	x/y/r: x/y/r-channel.
	Tmc: Mixing chamber temperatures.
	Tmct: MCT temperatures.
	Cmct: MCT capacitances.
	'''
	fo=open(filename)
	data=np.loadtxt(fo)
	f=data[:,0]*ftimes
	x=data[:,1]*xtimes
	y=data[:,2]*ytimes
	r=data[:,3]*rtimes
	Tmc=data[:,4]
	Tmct=data[:,5]
	Cmct=data[:,6]
	fo.close()
	return f,x,y,r,Tmc,Tmct,Cmct
#=======================================================================
def unity(f,rawdata):
	'''
	This is a simple toolkit which does not alter rawdata at all.
	Syntax:
	-------
	newdata=unity(f,rawdata)
	Parameters:
	-----------
	f: Dummy input, not used by program.
	rawdata: Input data.
	Returns:
	--------
	newdata: will be the same as rawdata
	'''
	return rawdata
#=======================================================================
def mknmrp0Header(length):
	'''
	Make header line for nmr fitting parameters. Assume 4 parameters [s0,T,f0,phase] for each FFT FID peak.
	e.g. if length==12, then output would be header=
	['s00','T0','f00','phase0','s01','T1','f01','phase1','s02','T2','f02','phase2']
	Syntax:
	-------
	header=mknmrp0Header(length)
	Parameters:
	-----------
	length: Length of the header wished to be made. Should be 4xN, N=0,1,2,... If length=4,5,6,7 will give the same results, length=8,9,10,11 will give the same results, etc..
	Returns:
	--------
	header: Generated header line.
	'''
	numpk=int(length/4) #numpk:number of peaks
	header=[]
	for i in range(0,numpk):
		ite=str(i)
		header+=['s0'+ite,'T'+ite,'f0'+ite,'phase'+ite]
	return header
#=======================================================================
def fswpFitLoad(filename,filepopt,header):
	'''
	Fetch optimized parameter for one FreqSweep file from fitting results saved in another file.
	Syntax:
	-------
	popt=fswpFitLoad(filename,filepopt,header)
	Parameters:
	-----------
	filename: str, FreqSweep file name.
	filepopt: str, file that records optimized parameters, assumed to use whitespace as delimiter.
	header: str list, header list corresponding to popt.
	Returns:
	--------
	popt: np.array, recorded popt for the file pointed to by filename.
	'''
	df=pd.read_csv(filepopt,delim_whitespace=True)
	r=df[df['Filename']==filename]
	condition=[(elem in header) for elem in df.columns]
	popt=r[r.columns[condition]].values[0]
	return popt
#=======================================================================
def nmrFitLoad(filenmr,filepopt):
	'''
	2019-10-18 15:18
	Fetch optimized parameter for one nmr file from fitting results saved in another file.
	Syntax:
	-------
	popt,zerofillnum=nmrFitLoad(filenmr,filepopt)
	Parameters:
	-----------
	filenmr: str, NMR file name.
	filepopt: str, file that records all optimized parameters. File assumed to use whitespace as delimiter.
	Returns:
	--------
	popt: np.array, recorded popt for the file pointed to by filenmr.
	zerofillnum: number of zerofilling points used to do this fit.
	'''
	df=pd.read_csv(filepopt,delim_whitespace=True)
	r=df[df['Filename']==filenmr]
	truncated=[elem[:-1:] for elem in r.columns] #rid last header symbol
	compare=[(elem=='s0')or(elem=='T')or(elem=='f0')or(elem=='phase') for elem in truncated]
	popt=r[r.columns[compare]].values[0]

	compareperr=[('perr' in elem) for elem in r.columns]
	perr=r[r.columns[compareperr]].values[0]
	
	zerofillnum=r['_zerofillnum'].values[0]
	return popt,perr,zerofillnum
#=======================================================================
def prepare_bounds(bounds,n):
	'''
	Prepare bounds, resize to match the longer of low&high bounds. This program is a direct copy from inside sicpy.optimize._lsq.lsq_linear.py.
	Syntax:
	-------
	lb,ub=prepare_bounds(bounds,n)
	Parameters:
	-----------
	bounds: [lb,ub] format, if lb is not a list but a number, it will be resized to n-dim long; the same for ub.
	n: size to be matched.
	Returns:
	--------
	lb,ub: resized lb,ub.
	'''
	lb,ub=[np.asarray(b) for b in bounds]
	if lb.ndim==0:
		lb=np.resize(lb,n)
	if ub.ndim==0:
		ub=np.resize(ub,n)
	return lb, ub
#=======================================================================
def range_condition(lb,ub,x):
	'''
	Build condition based on x values that lie within the range defined by lb and ub.
	Syntax:
	-------
	AndCond,OrCond=range_condition(lb,ub,x)
	Parameters:
	-----------
	lb: np.array, lower bound.
	ub: np.array, upper bound.
	x: x values based in which the range is chosen.
	Returns:
	--------
	AndCond: The strictest range condition, x[AndCond] is the intersection of all ranges.
	OrCond: The loosest range condition, x[OrCond] is the union of all ranges.
	'''
	AndCond=True
	OrCond=False
	for l,u in zip(lb,ub):
		condition=(x>=l)&(x<=u) # includes boundary points
		AndCond=AndCond & condition
		OrCond=OrCond | condition
	return AndCond,OrCond
#=======================================================================
def range_condition_series(lb,ub,x):
	'''
	Build condition based on x values that lie within the range between lb and ub.
	Syntax:
	-------
	AndCond,OrCond=range_condition_series(lb,ub,x)
	Parameters:
	-----------
	lb: np.array, lower bound.
	ub: np.array, upper bound.
	x: pandas.Series, x values based on which the range is chosen.
	Returns:
	--------
	AndCond: pandas.Series, the strictest range condition, x[AndCond] is the intersection of all ranges.
	OrCond: pandas.Series, he loosest range condition, x[OrCond] is the union of all ranges.
	'''
	AndCond=True
	OrCond=False
	for l,u in zip(lb,ub):
		condition1=x.between(l,u) # includes boundary points
		condition2=x.between(u,l) # allow to recognize reversed bounds
		condition=condition1 | condition2
		AndCond=AndCond & condition
		OrCond=OrCond | condition
	return AndCond,OrCond
#=======================================================================
def build_condition(bounds,x):
	'''
	Build conditions based on x from input bounds.
	Syntax:
	-------
	AndCond,OrCond=build_condition(bounds,x)
	Parameters:
	-----------
	bounds: [lb,ub] format, if lb is not a list but a number, it will be resized to n-dim long; the same for ub. 
	x: x values based on which the conditions are produced.
	Returns:
	--------
	AndCond: The strictest range condition, x[AndCond] is the intersection of all ranges.
	OrCond: The loosest range condition, x[OrCond] is the union of all ranges.
	'''
	n=max(np.asarray(bounds[0]).size,np.asarray(bounds[1]).size) #choose the longer one's dimension as n
	lb,ub=prepare_bounds(bounds,n)
	AndCond,OrCond=range_condition(lb,ub,x)
	return AndCond,OrCond
#=======================================================================
def build_condition_series(bounds,x):
	'''
	Build conditions based on x from input bounds.
	Syntax:
	-------
	AndCond,OrCond=build_condition_series(bounds,x)
	Parameters:
	-----------
	bounds: [lb,ub] format, if lb is not a list but a number, it will be resized to n-dim long; the same for ub. 
	x: pandas.Series, x values based on which the conditions are produced.
	Returns:
	--------
	AndCond: pandas.Series, the strictest range condition, x[AndCond] is the intersection of all ranges.
	OrCond: pandas.Seires, the loosest range condition, x[OrCond] is the union of all ranges.
	'''
	n=max(np.asarray(bounds[0]).size,np.asarray(bounds[1]).size) #choose the longer one's dimension as n
	lb,ub=prepare_bounds(bounds,n)
	AndCond,OrCond=range_condition_series(lb,ub,x)
	return AndCond,OrCond
#=======================================================================
def build_condition_dataframe(bounds,dataframe,colname):
	'''
	Check input dataframe under column name specified by colname, find the pieces fall between the range specified by frange.
	Syntax:
	-------
	AndCond,OrCond=build_condition_dataframe(bounds,dataframe,colname)
	Parameters:
	-----------
	bounds: (lowb,highb) format range, lowb is a single item, or lowb is a list of all the lowbounds, corresponding to some high bounds stored in highb. lowb==highb is allowed.
	dataframe: pandas.DataFrame type input log.
	colname: column name under which the frange is applied, and the conditions are built.
	Returns:
	--------
	AndCond: Boolean array. For each low/high-bounds pair, there is a range, AndCond is the intersection of all these ranges.
	OrCond: The union of all ranges.
	Notes:
	------
	This program raises error if the column specified by colname is not single-valued.
	Here, frange refers to file_range, but the program can be implemented for other columns as well.
	'''
	AndCond,OrCond=build_condition_series(bounds,dataframe[colname])
	return AndCond,OrCond
#=======================================================================
def partial_dataframe_mean(frange,dataframe,colname,droplabels=None,dropAxis=1,meanAxis=0):
	'''
	Pick part of a pandas.DataFrame, then calculate the numeric-only mean and standard deviation of the dataframe, and outputs a pandas.Series
	Syntax:
	-------
	pMean,pStd=partial_dataframe_mean(frange,dataframe,colname,droplabels=None,dropAxis=1,meanAxis=0)
	Parameters:
	-----------
	frange: (lowb,highb) format range, lowb is a single item, or lowb is a list of all the lowbounds, corresponding to some high bounds stored in high b. lowb==highb is allowed. The data within this range will be averaged.
	dataframe: pandas.DataFrame.
	colname: name of the column in which the frange exists.
	droplabels: single label or list-like, labels to be excluded from averaging.
	dropAxis: dataframe.drop axis, 1 means drop columns.
	meanAxis: dataframe.mean/std axis, 0 means average the column values.
	Returns:
	--------
	pMean: pandas.Series, mean of the chosen piece of dataframe, numeric only.
	pStd: pandas.Series, standard deviation of the chosen piece of dataframe, numeric only.
	'''
	_,OrCond=build_condition_dataframe(frange,dataframe,colname)
	piece=dataframe[OrCond]

	if droplabels is not None : #drop specific row/columns
		piece=piece.drop(droplabels,axis=dropAxis)
	
	pMean=piece.mean(axis=meanAxis,numeric_only=True)
	pStd=piece.std(axis=meanAxis,numeric_only=True)
	
	return pMean,pStd
#=======================================================================
def dataframe_cluster(dataframe,dropIndex=pd.Series({'':[]}).index):
	'''
	Pick the dataframe with columns not in dropIndex, then output the 1st row, the last row, and the sorted concatenation of the previous two.
	Syntax:
	-------
	clus1st,cluslast,clus_1st_last=dataframe_cluster(dataframe,dropIndex=pd.Series({'':[]}).index)
	Parameters:
	-----------
	dataframe: pandas.DataFrame.
	dropIndex: pandas.Index, default drops one element '', the column shared in this index will be dropped from output.
	Returns:
	--------
	clus1st: pandas.Series, 1st row of data after drop.
	cluslast: pandas.Series, last row of data after drop.
	clus_1st_last: pandas.Series, a sorted concatenation of clus1st and cluslast.
	'''
	condition=[(elem not in dropIndex.values) for elem in dataframe.columns.values]
	clusLabels=dataframe.columns[condition] #index of column names after excluding the ones specified in dropIndex, pandas.Series

	clus1st=dataframe[clusLabels].iloc[0,:] #1st row of piece with survived labels
	cluslast=dataframe[clusLabels].iloc[-1,:] #last row, pandas.Series
	clus1st.index=clusLabels.values+'_1st'
	cluslast.index=clusLabels.values+'_last'
	sortIndex=[]
	for (i,j) in zip(clusLabels.values+'_1st',clusLabels.values+'_last'):
		sortIndex+=[i,j]
	
	clus_1st_last=pd.concat([clus1st,cluslast])
	clus_1st_last=clus_1st_last.reindex(index=sortIndex)
	return clus1st,cluslast,clus_1st_last
#=======================================================================
def partial_dataframe_cluster(frange,dataframe,colname,dropIndex=pd.Series({'':[]}).index):
	'''
	Pick part of a pandas.Dataframe to form a piece, then apply dataframe_cluster on this piece.
	Syntax:
	-------
	clus1st,cluslast,clus_1st_last=partial_dataframe_cluster(frange,dataframe,colname,dropIndex=pd.Series({'':[]}).index)
	Parameters:
	-----------
	frange: (lowb,highb) format range, lowb is a single item, or lowb is a list of all the lowbounds, corresponding to some high bounds stored in high b. lowb==highb is allowed. The data within this range will be averaged.
	dataframe: pandas.DataFrame.
	colname: name of the column in which the frange exists.
	dropIndex: pandas.Index, default drops one element '', the column shared in this index will be dropped from output.
	Returns:
	--------
	clus1st: pandas.Series, 1st row of data after drop.
	cluslast: pandas.Series, last row of data after drop.
	clus_1st_last: pandas.Series, a sorted concatenation of clus1st and cluslast.
	'''
	_,OrCond=build_condition_dataframe(frange,dataframe,colname)
	piece=dataframe[OrCond]

	clus1st,cluslast,clus_1st_last=dataframe_cluster(piece,dropIndex=dropIndex)
	
	return clus1st,cluslast,clus_1st_last
#=======================================================================
def fswpVsNmr(device,filenums,fswpfilepopt=None,nmrfilepopt=None):
	'''
	Read specific rows from fitted parameter storage file, fswpfilepopt, that corresponds to input filename, then read nmrfilepopt for the rows associated with these sweeps.
	Syntax:
	-------
	fswppiece,nmrpiece,mergepiece=fswpVsNmr(device,filenums,fswpfilepopt=None,nmrfilepopt=None)
	Parameters:
	-----------
	device: str, device code.
	filenums: tuple, (filelow, filehigh) range, filelow and filehigh can both be lists, they must have the same length.
	fswp/nmrfilepopt: str, names of the files storing fit results, these files must use whitespace as delimiters.
	Returns:
	--------
	fswppiece: pandas.DataFrame, a piece of fswpfilepopt containing only info associated with the files specified by filenums.
	nmrppiece: pandas.DataFrame, a piece of nmrfilepopt containing only info associated with the fswp files in fswppiece.
	mergepiece: pandas.DataFrame, the merged dataframe of fswppiece and nmrpiece.
	Notes:
	------
	fswp/nmrfilepopt are not optional.
	'''
	#make filenames from device and filenums
	vfunc=np.vectorize(mkFilename)
	filenames=(vfunc(device,filenums[0]),vfunc(device,filenums[1]))
	
	fswplog=pd.read_csv(fswpfilepopt,delim_whitespace=True)
	nmrlog=pd.read_csv(nmrfilepopt,delim_whitespace=True)
	_,OrCond=build_condition_dataframe(filenames,fswplog,'Filename')
	fswppiece=fswplog[OrCond]
	nmrpiece=nmrlog[[elem in fswppiece['NMRFilename'].values for elem in nmrlog['NMRFilename']]]

	mergepiece=pd.merge(fswppiece,nmrpiece)
	return fswppiece,nmrpiece,mergepiece
#=======================================================================
def prepend_header(path,header):
	'''
	Prepend header content to existing file. The access and modify times of the file will be preserved. Extra '\r\n\n' is inserted between new header and the original content
	Syntax:
	-------
	prepend_header(path,header)
	Parameters:
	-----------
	path: path of the file to modify
	header: the header content to be prepended.
	Returns:
	--------
	None.
	'''
	epocha=os.path.getatime(path)
	epochm=os.path.getmtime(path)
	fo=open(path,mode='r+') # read and write
	content=fo.read()
	fo.seek(0) # move cursor to beginning of file
	fo.write(header+'\r\n\n'+content) # prepend header w/ empty newline
	fo.truncate() # necessary to keep out redundant date
	fo.close()
	os.utime(path,times=(epocha,epochm)) # restore access and modify time
	return None
#=======================================================================
def remove_header(path):
	'''
	Remove first line of a given file. The access and modify times of the file will be preserved.
	Syntax:
	-------
	remove_header(path)
	Parameters:
	-----------
	path: path of the file to modify.
	Returns:
	--------
	None.
	'''
	epocha=os.path.getatime(path)
	epochm=os.path.getmtime(path)
	with open(path,'r') as fin:
		data=fin.read().splitlines(True)
	with open(path,'w') as fout:
		fout.writelines(data[1:])
	os.utime(path,times=(epocha,epochm)) # restore access and modify time
	return None
#=======================================================================
def tcxR2T(R):
	'''
	CX-1080-CU-HT-20L thermometer's resistance to temperature conversion. The standard current is 11uA. Usually a measurement includes measurements at both +11uA and -11uA, and then average between two R values.
	Syntax:
	-------
	T=tcxR2T(R)
	Parameters:
	-----------
	R: a number, the resistance value of the thermometer in Ohm.
	Returns:
	--------
	T: a number, the measured temperature in Kelvin.
	'''
	ZL=2.19190615742
	ZU=3.06365212051
	a0=185.729684
	a1=-119.272440
	a2=21.243200
	a3=-3.132976
	a4=0.523229
	a5=-0.092608
	a6=0.015490
	a7=-0.003148
	A=[a0,a1,a2,a3,a4,a5,a6,a7]

	Z=np.log10(R)
	k=((Z-ZL)-(ZU-Z))/(ZU-ZL)
	T=0
	for i in range(0,8):
		T+=A[i]*np.cos(i*np.arccos(k))
	return T
#=======================================================================
def dt670v2t(V):
    '''
    Lakeshore DT-670 silicon diode voltage to temperature conversion. The standard current is 10uA. For the polynomial fitting results used here, I used
    T=np.array([  1.4 ,   1.5 ,   1.6 ,   1.7 ,   1.8 ,   1.9 ,   2.  ,   2.1 ,
         2.2 ,   2.3 ,   2.4 ,   2.5 ,   2.6 ,   2.7 ,   2.8 ,   2.9 ,
         3.  ,   3.1 ,   3.2 ,   3.3 ,   3.4 ,   3.5 ,   3.6 ,   3.7 ,
         3.8 ,   3.9 ,   4.  ,   4.2 ,   4.4 ,   4.6 ,   4.8 ,   5.  ,
         5.2 ,   5.4 ,   5.6 ,   5.8 ,   6.  ,   6.5 ,   7.  ,   7.5 ,
         8.  ,   8.5 ,   9.  ,   9.5 ,  10.  ,  10.5 ,  11.  ,  11.5 ,
        12.  ,  12.5 ,  13.  ,  13.5 ,  14.  ,  14.5 ,  15.  ,  15.5 ,
        16.  ,  16.5 ,  17.  ,  17.5 ,  18.  ,  18.5 ,  19.  ,  19.5 ,
        20.  ,  21.  ,  22.  ,  23.  ,  24.  ,  25.  ,  26.  ,  27.  ,
        28.  ,  29.  ,  30.  ,  31.  ,  32.  ,  33.  ,  34.  ,  35.  ,
        36.  ,  37.  ,  38.  ,  39.  ,  40.  ,  42.  ,  44.  ,  46.  ,
        48.  ,  50.  ,  52.  ,  54.  ,  56.  ,  58.  ,  60.  ,  65.  ,
        70.  ,  75.  ,  77.35,  80.  ,  85.  ,  90.  , 100.  , 110.  ,
       120.  , 130.  , 140.  , 150.  , 160.  , 170.  , 180.  , 190.  ,
       200.  , 210.  , 220.  , 230.  , 240.  , 250.  , 260.  , 270.  ,
       273.  , 280.  , 290.  , 300.  , 310.  , 320.  , 330.  , 340.  ,
       350.  , 360.  , 370.  , 380.  , 390.  , 400.  , 410.  , 420.  ,
       430.  , 440.  , 450.  , 460.  , 470.  , 480.  , 490.  , 500.  ])
    V=[1.644290,1.642990,1.641570,1.640030,1.638370,1.636600,1.634720,1.632740,1.630670,1.628520,1.626290,1.624000,1.621660,1.619280,1.616870,1.614450,1.612000,1.609510,1.606970,1.604380,1.601730,1.599020,1.596260,1.59344,1.59057,1.58764,1.58465,1.57848,1.57202,1.56533,1.55845,1.55145,1.54436,1.53721,1.53000,1.52273,1.51541,1.49698,1.47868,1.46086,1.44374,1.42747,1.41207,1.39751,1.38373,1.37065,1.35820,1.34632,1.33499,1.32416,1.31381,1.30390,1.29439,1.28526,1.27645,1.26794,1.25967,1.25161,1.24372,1.23596,1.22830,1.22070,1.21311,1.20548,1.197748,1.181548,1.162797,1.140817,1.125923,1.119448,1.115658,1.112810,1.110421,1.108261,1.106244,1.104324,1.102476,1.100681,1.098930,1.097216,1.095534,1.093878,1.092244,1.090627,1.089024,1.085842,1.082669,1.079492,1.076303,1.073099,1.069881,1.066650,1.063403,1.060141,1.056862,1.048584,1.040183,1.031651,1.027594,1.022984,1.014181,1.005244,0.986974,0.968209,0.949000,0.929390,0.909416,0.889114,0.868518,0.847659,0.826560,0.805242,0.783720,0.762007,0.740115,0.718054,0.695834,0.673462,0.650949,0.628302,0.621141,0.605528,0.582637,0.559639,0.536542,0.513361,0.490106,0.466760,0.443371,0.419960,0.396503,0.373002,0.349453,0.325839,0.302161,0.278416,0.254592,0.230697,0.206758,0.182832,0.159010,0.135480,0.112553,0.090681]
    p1 is obtained from 8th order polynomial fit of the calibration data T<=23K.
    p2 is obtained from 5th order polynomial fit of the calibration data 23K<=T<=30K.
    p3 is obtained from 11th order polynomial fit of the calibration data T>=30K.
    
    Syntax:
    -------
    T=dt670v2t(V)
    Parameters:
    -----------
    V: (array/list of) float, the voltage value across the diode when provided 10uA current.
    Returns:
    --------
    T: (array/list of) float, the measured temperature in Kelvin. If the temperature is out of the 1.4-500K range, NaN is returned.
    '''

    p1=np.array([-20565.913782268446, 228730.4194910006, -1119517.6988040218, 3150831.1077316683, -5578744.951300982, 6363503.260132671, -4565985.813483451, 1883378.9740621012, -341630.3678481866])
    p2=np.array([386892815.6121058, -2175204494.451572, 4891470786.059078, -5499437047.54732, 3091262041.4756904, -694994849.5481056]) 
    p3=np.array([134033.55576023352, -842414.0610855379, 2315252.7422276624, -3654248.7357260347, 3656374.766135586, -2414328.5685193813, 1060660.0291558076, -304469.54933367047, 54273.374297208655, -5381.362845641886, -195.99048275081765, 536.779755950008])
    
    def v2t(V): # single value V to T conversion
        if V>=1.140817:
            T=np.polynomial.polynomial.polyval(V,p1[-1::-1])
        elif V<=1.106244:
            T=np.polynomial.polynomial.polyval(V,p3[-1::-1])
        else:
            T=np.polynomial.polynomial.polyval(V,p2[-1::-1])

        if T<1.4 or T>500: # out of range
            return np.nan
        else:
            return T
    
    return np.vectorize(v2t)(V) # allows for array input/output
#=======================================================================
def setra206V2P(V):
	'''
	Setra 206 pressure gauge voltage to pressure conversion. The supply is 24VDC. The manufacturer calibration data is used:
	P=[0.7375,40.7079,93.3524,144.7748,195.4766,247.7697,299.2034,350.5528,401.6060,453.5418,505.1436] (psig)
	V=[0.25,0.6379,1.1479,1.6467,2.1389,2.6459,3.1450,3.6438,4.1396,4.6457,5.1497] (VDC)	
	The formula used here is obtained from a linear regression of the calibration data.
	Syntax:
	-------
	P=setra206V2P(V)
	Parameters:
	-----------
	V: a number, DC voltage output of the setra206 gauge.
	Returns:
	--------
	P: float, pressure measured by the gauge.
	'''
	slope=102.98806383700568
	intercept=-24.860760933856199
	P=slope*V+intercept
	return P
#=======================================================================
def setra206P2V(P):
	'''
	Setra 206 pressure gauge pressure to voltage conversion. The supply is 24VDC. This is the exact reverse of setra206V2P using the same manufacturer calibration data.
	Syntax:
	-------
	V=setra206P2V(P)
	Parameters:
	-----------
	P: a number, pressure measured by the gauge.
	Returns:
	--------
	V: float, DC voltage output of the setra206 gauge.
	'''
	slope=102.98806383700568
	intercept=-24.860760933856199
	V=(P-intercept)/slope
	return V
#=======================================================================
def jnp_zeros(m,n):
	'''
	Compute zeros of integer-order Bessel function derivative Jn’(x)
	This program defines a '0th' extremum for the Bessel function. Utility.jnp_zeros(0,0) will return 0.
	Syntax:
	-------
	jnp0=jnp_zeros(m,n)
	Parameters:
	-----------
	m: interger, integer-order of the Bessel function.
	n: interger, n stands for the 'nth' extremum for the Bessel function.
	Returns:
	--------
	jnp0: float, the nth extremum for the Bessel function.
	'''
	if m==0 and n==0:
		return 0
	else:
		return scipy.special.jnp_zeros(m,n)[-1]
#=======================================================================
def cylCav_f0(L,R,c,l,m,n):
	'''
	Calculate the acoustic wave eigen frequencies inside a cylindrical cavity. The amplitude of the oscillating pressure inside the cavity at location (r,theta, z) is P_{lmn}=A_{lmn}J_m(k_{mn}r)\cos(m\ theta+\gamma_{lmn})\cos(k_{zl}z). Check my notes for more information. The same notation is used.
	Syntax:
	-------
	flmn=cylCav_f0(L,R,c,l,m,n)
	Parameters:
	-----------
	L: float, length of the cavity.
	R: float, radius of the cavity.
	c: float, speed of sound of the medium filling the cavity.
	l,m,n: integers, integer-order numbers for the wave. Check my notes
	'''
	kzl=l*np.pi/L
	kmn=jnp_zeros(m,n)/R
	flmn=1/2/np.pi*c*np.sqrt(kmn**2+kzl**2)
	return flmn
#=======================================================================
