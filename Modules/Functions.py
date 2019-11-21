'''
Miscellaneous use functions.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy import fftpack

import FuncLib
import Plotting
import Utility as utl
import readLog
#=======================================================================
def paramsize(function):
	'''
	Syntax:
	-------
	ps=paramsiz(function)
	Parameters:
	-----------
	function: function whose parameter number needs to be assessed.
	Returns:
	--------
	ps: assumes form function(x,p0,p1,...,pN), p must be a list, returns len(p)=N+1.
	'''
	p=[1.] #will lose outer brackets when parsed as *p into func
	for i in range(10000): #run 10000 times max
		try:
			function(1.,*p);
			break

		except TypeError:
			p.append(1.)
	return len(p)
#=======================================================================
def assemble(funcs,folds):
	'''
	Do summation of functions.
	Syntax:
	-------
	newfunc=assemble(funcs,folds)
	Parameters:
	-----------
	funcs: functions list, these functions will be added up.
	folds: folds list, must have len(folds)==len(funcs).
	Returns:
	--------
	newfunc: function type, mathematically speaking:
	newfunc=folds[0]*funcs[0]+folds[1]*funcs[1]+...+folds[-1]*funcs[-1].
	newfunc(f,*p) takes 1+len(p) inputs, f the independent var representing the first input of all functions in funcs, and p containing all other additional parameters required by functions in funcs in an orderly fashion.
	e.g. newfunc=Functions.assemble([FuncLib.lrtzX,FuncLib.bgInv2],[1,1]) will give newfunc(f,*p) where p=[A,d,f0,c2].
	'''
	def newFunc(f,*p): #do summation to construct a new function
		f=np.array(f,float)
		p=np.array(p,float)
		iter=0
		output=0.
		for fold,func in zip(folds,funcs):
			ps=paramsize(func) # this many parameters need to be parsed
			output+=fold*func(f,*p[iter:iter+ps])#add fold*func to output
			iter+=ps
		return output
	return newFunc #return the function object
#=======================================================================
def assembleShare(func1,func2,sharenum):
	'''
	Do summation of functions func1 and func2 who share N=sharenum parameters.
	Syntax:
	-------
	newfunc=assembleShare(func1,func2,sharenum)
	Parameters:
	-----------
	func1,func2: functions to be added up, both accepting input format (f,*p).
	sharenum: the public parameters to be shared between func1&2.
	Returns:
	--------
	newfunc: a function, output=newfunc(f,p) assigns p into p1&p2 with shared parameters parsed in front, then returns a concatenated array of the form output=list(output1)+list(output2). Note that len(f)=len(output1)=len(output2)=1/2*len(output).
	'''
	def newFunc(f,*p):
		f=np.array(f,float)
		p=np.array(p,float)
		pnum1=paramsize(func1)
		pnum2=paramsize(func2)
		p1=p[0:pnum1]
		p2=np.concatenate((p[0:sharenum],p[pnum1:pnum1+pnum2-sharenum]))
		
		out1=func1(f,*p1)
		out2=func2(f,*p2)
		if type(out1) is not np.ndarray: #if out1 is a single number
			out1=[out1] #avoid 0-d concatenate error
			out2=[out2]
		
		return np.concatenate((np.array(out1),np.array(out2)))
	
	return newFunc
#=======================================================================
def paramUnfold(popt,funcs1,folds1,funcs2,folds2,sharenum):
	'''
	Translate popt based on funcs1,folds1,funcs2,folds2 to remove effect of folds1&2 in popt.
	The new popt corresponds to fitting result from folds1&2 contain only number 1s.
	Syntax:
	-------
	popt,popt1,popt2=paramUlfold(popt,funcs1,folds1,funcs2,folds2,sharenum)
	Parameters:
	-----------
	popt: fitting parameters.
	funcs1&2: function list 1&2.
	folds1&2: fold list 1&2.
	sharenum: number of parameters shared by funcs1&2.
	Results:
	--------
	popt: updated popt without folding.
	popt1&2: seperated popt, popt1&2 are fitted parameters for funcs1&2.
	Note:
	-----
	This function will not work for FuncLib.lrtzRR.
	'''
	pnum1=paramsize(assemble(funcs1,folds1))
	popt1=popt[0:pnum1]
	popt2=np.concatenate((popt[0:sharenum],popt[pnum1:]))
	i=0
	for func,fold in zip(funcs1,folds1):
		popt1[i]*=fold
		i+=paramsize(func)
	
	i=0
	for func,fold in zip(funcs2,folds2):
		popt2[i]*=fold
		i+=paramsize(func)
	
	popt=np.concatenate((popt1,popt2[sharenum:]))
	return popt,popt1,popt2
#=======================================================================
def paramGuess(data,fitmode='noCorrect'):
	'''
	2017-06-22 17:52
	Guess A,d,f0 from given data x-channel.
	Syntax:
	-------
	[A,d,f0]=paramGuess(data,fitmode='string')
	Parameters:
	-----------
	data: sweep.freqSweep class, with .f/.r/.gr attributes.
	fitmode: fit mode, x=data.gr instead of .r if 'g' in fitmode, default is 'noCorrect'.
	Returns:
	--------
	[A,d,f0]: list, contains A,d,f0 from lrtz model.
	'''
	# deprecated version for FreqSweep.FreqSweep
	#if 'g' in fitmode:
	#	x=data.gx
	#else:
	#	x=data.x

	#peak=x.max()
	#f0=data.f[x==peak][0] #resonance freq
	#condition=data.x<=data.x.max()/2 #at x no larger than max
	#f1=data.f[condition&(data.f<f0)].max() # FWHM lower freq
	#f2=data.f[condition&(data.f>f0)].min() # FWHM higher freq
	
	if 'g' in fitmode:
		r=data.gr
	else:
		r=data.r
	
	peak=r.max() # peak height
	peak_idx=r.idxmax() # the index of the maximum r point
	f0=data.f[peak_idx] # resonance freq
	condition=data.r>=data.r.max()/2 # find higher half of peak
	fwhm_range=data.f[condition] # full-width-half-maximum range
	f1=fwhm_range.min()
	f2=fwhm_range.max()
	d=f2-f1 #width
	A=4*np.pi**2*peak*d*f0 # peak amplitude
	return [A,d,f0]
#=======================================================================
def lrtz1fit(data,fitmode,funcs,folds,p0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf)):
	'''
	Load data based on fitmode, and fit according to constructed model from funcs,folds.
	Syntax:
	-------
	popt,pcov,res=lrtz1fit(data,fitmode,funcs,folds,p0[,bounds=(#,#)])
	Parameters:
	-----------
	data: FreqSweep class object with attributes .f/.x/.y/.r/.gx/.gy/.gr.
	fitmode: if includes 'g', fit gx/gy/gr, otherwise fit x/y/r against f.
	funcs: fitting function model list.
	folds: fitting function folds list.
	p0: initial guessed parameters.
	frange: (lower,higher) bounds of frequency range in Hz.
	bounds: fitting parameter bounds, check scipy.optimize.curve_fit for details.
	Default values are: frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf).
	Returns:
	--------
	popt: optimized fitting parameters.
	pcov: the estimated covariance of popt.
	res: residual = model(f,popt)-measurement
	'''
	fitmode=fitmode.lower() #make input case insensitive
	fitmode=fitmode.replace('f','') #we always use frequency as x-var
	if 'g' in fitmode: #determine if gain correct fit is required
		fitmode=fitmode.replace('g','') #remove 'g'
		yvar=fitmode[0].replace(fitmode[0],'g'+fitmode[0]) #append 'g' in front
	else:
		yvar=fitmode[0]
	
	x=data.f
	y=eval('data.'+yvar)
	condition=(x>=frange[0])&(x<=frange[1])
	x=x[condition]
	y=y[condition]
	
	popt,pcov=scipy.optimize.curve_fit(assemble(funcs,folds),x,y,p0=p0,bounds=bounds)#do fit
	res=assemble(funcs,folds)(x,*popt)-y
	return popt,pcov,res
#=======================================================================
def lrtz1simfit(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
	'''
	Simultaneously fit x-&y-channels, with x in front. Plot fitted curve if demanded.
	Syntax:
	-------
	popt,pcov,perr,res,popt1,popt2,fig,axes,lines=lrtz1simfit(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0[,frange=(-inf,inf),bounds=(-inf,inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
	Parameters:
	-----------
	data: FreqSweep class data object.
	fitmode: only difference is if it contains 'g' or not.
	funcs1&2: function lists of models for simultaneous fitting.
	folds1&2: function fold lists, corresponding to terms in funcs1&2.
	sharenum: number of parameters shared by funcs1&2.
	p0: initial parameters guess.
	frange: frequency range (low,high) bounds. low/high can be a list or a single items.
	bounds: parameters bounds, check scipy.optimize.curve_fit input.
	pltflag: if non-zero, will plot fitted curves for comparison.
	figsize: figure size.
	wspace,hspace: width/horizontal spacing between subplots.
	markersize,linewidth: matplotlib.pyplot.plot inputs.
	legloc: legend location.
	bbox_to_anchor: legend anchor point.
	legsize: legend font size.
	Returns:
	--------
	popt: fitted parameters, folds1&2 influence removed.
	pcov: 2d array, the estimated covariance of popt.
	perr: standard deviation associated with popt.
	res: residual = calculated values from popt - data values.
	popt1&2: popt separated into two parts corresponding to funcs1&2.
	fig: figure handle.
	axes: 2x2 axes handles array.
	lines: lines output from Plotting.fitChecksim().
	'''
	if 'g' in fitmode: #determine if gain correct fit is required
		y1=data.gx
		y2=data.gy
	else:
		y1=data.x
		y2=data.y
	
	_,OrCond=utl.build_condition(frange,data.f)
	x=data.f[OrCond] # put data.f within frange into x
	y1=y1[OrCond] # truncate y1,y2 according to x, too
	y2=y2[OrCond]

	y=np.concatenate((y1,y2)) #concatenate two channels to create signal
	model1=assemble(funcs1,folds1) #create x model
	model2=assemble(funcs2,folds2) #create y model
	fitmodel=assembleShare(model1,model2,sharenum)

	popt,pcov=scipy.optimize.curve_fit(fitmodel,x,y,p0=p0,bounds=bounds) #do fit, len(f)=1/2*len(y)
	perr=np.sqrt(np.diag(pcov)) #standard deviation
	res=fitmodel(x,*popt)-y #calculate residual before update popt

	popt,popt1,popt2=paramUnfold(popt,funcs1,folds1,funcs2,folds2,sharenum)

	if pltflag: #plot after update popt
		fig,axes=plt.subplots(2,2,figsize=figsize)
		fig.subplots_adjust(wspace=wspace,hspace=hspace)
		lines=Plotting.fitCheckSim(axes,data,fitmode,funcs1,funcs2,sharenum,popt1,popt2,res,frange=frange,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
		return popt,pcov,perr,res,popt1,popt2,fig,axes,lines

	return popt,pcov,perr,res,popt1,popt2
#=======================================================================
def lrtz_1simfit(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
	'''
	Simultaneously fit x-&y-channels, with x in front. Plot fitted curve if demanded. Function designed for sweep.py.
	Syntax:
	-------
	popt,pcov,perr,res,popt1,popt2[,fig,axes,lines]=lrtz_1simfit(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0[,frange=(-inf,inf),bounds=(-inf,inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
	Parameters:
	-----------
	data: sweep class data object.
	fitmode: only difference is if it contains 'g' or not.
	funcs1&2: function lists of models for simultaneous fitting.
	folds1&2: function fold lists, corresponding to terms in funcs1&2.
	sharenum: number of parameters shared by funcs1&2.
	p0: initial parameters guess.
	frange: frequency range (low,high) bounds. low/high can be a list or a single items.
	bounds: parameters bounds, check scipy.optimize.curve_fit input.
	pltflag: if non-zero, will plot fitted curves for comparison.
	figsize: figure size.
	wspace,hspace: width/horizontal spacing between subplots.
	markersize,linewidth: matplotlib.pyplot.plot inputs.
	legloc: legend location.
	bbox_to_anchor: legend anchor point.
	legsize: legend font size.
	Returns:
	--------
	popt: fitted parameters, folds1&2 influence removed.
	pcov: 2d array, the estimated covariance of popt.
	perr: standard deviation associated with popt.
	res: residual = calculated values from popt - data values.
	popt1&2: popt separated into two parts corresponding to funcs1&2.
	+++
	if pltflag=True:
	fig: figure handle.
	axes: 2x2 axes handles array.
	lines: lines output from Plotting.fitCheck_1sim().
	'''
	if 'g' in fitmode: #determine if gain correct fit is required
		y1=data.gx
		y2=data.gy
	else:
		y1=data.x
		y2=data.y
	
	_,OrCond=utl.build_condition_series(frange,data.f)
	x=data.f[OrCond].values # put data.f within frange into x
	y1=y1[OrCond].values # truncate y1,y2 according to x, too
	y2=y2[OrCond].values

	y=np.concatenate((y1,y2)) #concatenate two channels to create signal
	model1=assemble(funcs1,folds1) #create x model
	model2=assemble(funcs2,folds2) #create y model
	fitmodel=assembleShare(model1,model2,sharenum)

	popt,pcov=scipy.optimize.curve_fit(fitmodel,x,y,p0=p0,bounds=bounds) #do fit, len(f)=1/2*len(y)
	perr=np.sqrt(np.diag(pcov)) #standard deviation
	res=fitmodel(x,*popt)-y #calculate residual before update popt

	popt,popt1,popt2=paramUnfold(popt,funcs1,folds1,funcs2,folds2,sharenum)

	if pltflag: #plot after update popt
		fig,axes=plt.subplots(2,2,figsize=figsize)
		fig.subplots_adjust(wspace=wspace,hspace=hspace)
		lines=Plotting.fitCheck_1sim(axes,data,fitmode,funcs1,funcs2,sharenum,popt1,popt2,res,frange=frange,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
		return popt,pcov,perr,res,popt1,popt2,fig,axes,lines

	return popt,pcov,perr,res,popt1,popt2
#=======================================================================
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    '''
	Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    '''
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
#=======================================================================
def nmr1simfit(data,p0,window_size,order,sfrange=(-np.inf,np.inf),deriv=0,rate=1,bounds=(-np.inf,np.inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8):
	'''
	DEPRECATED
	Smooth FFT FID and then fit to several peaks.
	Syntax:
	-------
	popt,pcov,perr=nmr1simfit(data,p0,window_size,order[,sfrange=(-np.inf,np.inf),deriv=0,rate=1,bounds=(-np.inf,np.inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8])
	Parameters:
	-----------
	p0: Initial fitting parameters, format is [s01,T1,f01,phase1,s02,T2,f02,phase2,...], length=4xN.
	window_size,order,deriv,rate: savitzky_golay smooth method input parameters.
	sfrange: Frequency range to perform smoothing, boundary included.
	bounds: Fitting parameter bounds, for scipy.optimize.curve_fit.
	pltflag: plot flag.
	figsize,wspace,hspace: figure and subplots spacing settings.
	marker,markersize,linewidth: Fitted curve plot settings.
	legloc,bbox_to_anchor,legsize: legend position and size.
	Returns:
	--------
	popt: Fitting parameters optimized from p0.
	pcov: Covariance output from scipy.optimize.curve_fit.
	perr: Standard deviation associated with popt.
	'''
	zerofillnum=data.zerofillnum
	snmr=data.smooth(sfrange,window_size=window_size,order=order,deriv=deriv,rate=rate)
	p0=np.array(p0)
	#numpk=int(p0.size/4) #number of peaks in FFT, 4=([s0,T,f0,phase])
	
	#create function that takes only f and p as inputs
	#create function calculate real part of FuncLib.FID0fft combined from several peaks
	def newr(f,*p):
		return FuncLib.FID0ffts(f,*p,zerofillnum=zerofillnum).real
	#create function calculate imaginary part of FuncLib.FID0fft combined from several peaks	
	def newi(f,*p):
		return FuncLib.FID0ffts(f,*p,zerofillnum=zerofillnum).imag
	#create function calculate real and imaginary parts simultaneously, output 1-D array with [real,imag] format
	def new(f,*p):
		return np.append(newr(f,*p),newi(f,*p))
	
	y=np.append(snmr.real,snmr.imag) #fit to this
	popt,pcov=scipy.optimize.curve_fit(new,data.f,y,p0=p0,bounds=bounds) # do fit
	perr=np.sqrt(np.diag(pcov))
#--------------------------------plot-----------------------------------
	if pltflag:
		fig,axes=plt.subplots(1,3,figsize=figsize)
		fig.subplots_adjust(wspace=wspace,hspace=hspace)
		lines=Plotting.fitChecknmr(axes,data,popt,marker=marker,markersize=markersize,linewidth=linewidth,bbox_to_anchor=(0,1),legloc='lower left',legsize=8)
		lines[0][0].set_zorder(1) #plot FID at bottom
		lines[0][1].set_zorder(10) #plot fitted FID on top
		axes[0].plot(data.t0fill,fftpack.ifft(snmr).real,marker='.',markersize=1,linewidth=0,color='greenyellow',zorder=9) #show smoothed FID, beneath fitted FID line.
		return popt,pcov,perr,fig,axes,lines

	return popt,pcov,perr
#=======================================================================
def nmr_1simfit(data,p0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8):
	'''
	2019-09-25 15:45
	Smooth FFT FID and then fit to several peaks for nmr.nmr object class.
	Syntax:
	-------
	popt,pcov,perr=nmr_1simfit(data,p0[,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8])
	Parameters:
	-----------
	p0: list; initial fitting parameters, format is [s01,T1,f01,phase1,s02,T2,f02,phase2,...], length=4xN.
	frange: (lb,ub); lower and upper bound of the frequency range for fitting.
	bounds: Fitting parameter bounds, for scipy.optimize.curve_fit.
	pltflag: plot flag.
	figsize,wspace,hspace: figure and subplots spacing settings.
	marker,markersize,linewidth: Fitted curve plot settings.
	legloc,bbox_to_anchor,legsize: legend position and size.
	Returns:
	--------
	popt: Fitting parameters optimized from p0.
	pcov: Covariance output from scipy.optimize.curve_fit.
	perr: Standard deviation associated with popt.
	'''
	p0=np.asarray(p0)
	_,OrCond=utl.build_condition(frange,data._f0fill) # prepare to isolate data in the frange only

	#create function that takes only f and p as inputs
	#create function calculate real part of FuncLib.FID0fft combined from several peaks
	def newr(f,*p):
		return FuncLib.FID0ffts(f,*p,zerofillnum=data._zerofillnum).real[OrCond]
	#create function calculate imaginary part of FuncLib.FID0fft combined from several peaks	
	def newi(f,*p):
		return FuncLib.FID0ffts(f,*p,zerofillnum=data._zerofillnum).imag[OrCond]
	#create function calculate real and imaginary parts simultaneously, output 1-D array with [real,imag] format
	def new(f,*p):
		return np.append(newr(f,*p),newi(f,*p))

	y=np.append(data._fftnmr0fill[OrCond].real,data._fftnmr0fill[OrCond].imag) #fit to data in this frange
	popt,pcov=scipy.optimize.curve_fit(new,data._f,y,p0=p0,bounds=bounds) # do fit
	perr=np.sqrt(np.diag(pcov))
#--------------------------------plot-----------------------------------
	if pltflag:
		fig,axes=plt.subplots(1,3,figsize=figsize)
		fig.subplots_adjust(wspace=wspace,hspace=hspace)
		lines=Plotting.fitCheck_nmr(axes,data,popt,marker=marker,markersize=markersize,linewidth=linewidth,bbox_to_anchor=(0,1),legloc='lower left',legsize=8)
		lines[0][0].set_zorder(1) #plot FID at bottom
		lines[0][1].set_zorder(10) #plot fitted FID on top
		return popt,pcov,perr,fig,axes,lines

	return popt,pcov,perr
#=======================================================================

