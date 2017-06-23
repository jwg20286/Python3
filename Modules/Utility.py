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
	branch: the melting curve branchin in which the solution is searched for. 'low' is below 315.23959351mK, which is the lowest pressure point on the melting curve, 'high' is above this point.
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
def gainCorrect_20mVrms(f,rawdata):
	'''
	2017-06-23 11:09
	Use the frequency-dependent Gain of demodulation(SR7124, see Utility.gainVsF1) to correct and convert the measurements of the following lock-in.
	Sensitivity=20mVrms
	1/0.016=2.5/(2*0.02)

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
	newdata=0.016*rawdata/gainVsF1(f)
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
	f: the modulaing frequency (w2/2pi) in Hz in 1-40kHz range,	f list/tuple/numpy.ndarray will be converted to numpy.ndarray of dtype=float
	Returns:
	--------
	gain: the ratio of input (Ri*cos(w1t)cos(w2t) to output (Ro*cos(w2t)).
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
	r=df[df['NMRFilename']==filenmr]
	truncated=[elem[:-1:] for elem in r.columns] #rid last header symbol
	compare=[(elem=='s0')or(elem=='T')or(elem=='f0')or(elem=='phase') for elem in truncated]
	popt=r[r.columns[compare]].values[0]

	compareperr=[('perr' in elem) for elem in r.columns]
	perr=r[r.columns[compareperr]].values[0]
	
	zerofillnum=r['zerofillnum'].values[0]
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
		condition=x.between(l,u) # includes boundary points
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
def build_condition_dataframe(frange,dataframe,colname):
	'''
	Check input dataframe under column name specified by colname, find the pieces fall between the range specified by frange.
	Syntax:
	-------
	AndCond,OrCond=build_condition_dataframe(frange,dataframe,colname)
	Parameters:
	-----------
	frange: (lowb,highb) format range, lowb is a single item, or lowb is a list of all the lowbounds, corresponding to some high bounds stored in highb. lowb==highb is allowed.
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
	n=max(np.asarray(frange[0]).size,np.asarray(frange[1]).size)
	lb,ub=prepare_bounds(frange,n)
	AndCond=True
	OrCond=False
	for l,u in zip(lb,ub):
		condition1=dataframe.index>=dataframe[dataframe[colname]==l].index.tolist()[0] #choose first if multiple matches are found as lower bound
		condition2=dataframe.index<=dataframe[dataframe[colname]==u].index.tolist()[-1] #choose last if multiple matches are found as upper bound
		condition=condition1 & condition2 #select the piece between lower and upper bounds
		AndCond=AndCond & condition
		OrCond=OrCond | condition
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

