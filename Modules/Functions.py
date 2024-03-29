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
def fixParam(func, fix_index, fix_params, other_params):
	'''
	Create a function similar to func, but requires fewer parameters. It only requires other_params and fix_params are fixed, while func requires both to be variable.
	Syntax:
	-------
	func_fixParam = fixParam(func, fix_index, fix_params, other_params).
	Parameters:
	-----------
	func: function to be replaced.
	fix_index: int, slice or sequence of ints,
		    Object that defines the index or indices before which values is inserted. Check numpy.insert for more info.
	fix_params: array_like,
		    Values to insert into other_params. 
	other_params: array_like,
		    Input array.
	Returns:
	--------
	func_fixParam: function that replaces func. It requires fewer parameters than func. Instead of requiring params, it only requires other_params, while fix_params are given fixed values.
	Known issues:
	-------------
	If len(fix_index)>1, paramsize doesn't work on func_fixParam.
	'''
	other_params = np.asarray(other_params)
    
	def func_fixParam(f, *other_params):
	    param = np.insert(other_params, fix_index, fix_params) # combine fix_params and other_params to make params suitable for func.
	    return func(f, *param) # do the same job as func using param.

	return func_fixParam
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
def lrtz_1simfit(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,mainsize=None,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
	'''
	Simultaneously fit x-&y-channels, with x in front. Plot fitted curve if demanded. Function designed for sweep.py.
	Syntax:
	-------
	popt,pcov,perr,res,popt1,popt2[,fig,axes,lines]=lrtz_1simfit(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0[,frange=(-inf,inf),bounds=(-inf,inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
	Parameters:
	-----------
	data: sweep.freqSweep class data object.
	fitmode: str, 'g', 'n', 'gn', or none of the aforementioned, determines how to preprocess x- and y-ch data.
	funcs1&2: function lists of models for simultaneous fitting.
	folds1&2: function fold lists, corresponding to terms in funcs1&2.
	sharenum: number of parameters shared by funcs1&2.
	p0: initial parameters guess.
	frange: frequency range (low,high) bounds. low/high can be a list or a single items.
	bounds: parameters bounds, check scipy.optimize.curve_fit input.
	pltflag: if non-zero, will plot fitted curves for comparison.
	mainsize: int, the number of parameters representing the actual signal, the rest are viewed as backgrounds; if none, set mainsize=sharenum.
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
	if fitmode == 'g': #determine if gain correct fit is required
		y1=data.gx
		y2=data.gy
	elif fitmode == 'n':
		y1=data.nx
		y2=data.ny
	elif fitmode == 'gn':
		y1=data.gnx
		y2=data.gny
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
		lines=Plotting.fitCheck_1sim(axes,data,fitmode,funcs1,funcs2,sharenum,popt1,popt2,res,mainsize=mainsize,frange=frange,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
		return popt,pcov,perr,res,popt1,popt2,fig,axes,lines

	return popt,pcov,perr,res,popt1,popt2
#=======================================================================
def lrtz_1simfit_fixParam(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,fix_index,fix_param,p0,mainsize=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
        '''
	Simultaneously fit x-&y-channels with some parameters fixed to constant values, with x in front. Plot fitted curve if demanded. Function designed for sweep.py.
	Syntax:
	-------
	popt,pcov,perr,res,popt1,popt2[,fig,axes,lines]=lrtz_1simfit_fixParam(data,fitmode,funcs1,folds1,funcs2,folds2,sharenum,fix_index,fix_param,p0[,frange=(-inf,inf),bounds=(-inf,inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
	Parameters:
	-----------
	data: sweep.freqSweep class data object.
	fitmode: str, 'g', 'n', 'gn', or none of the aforementioned, determines how to preprocess x- and y-ch data.
	funcs1&2: function lists of models for simultaneous fitting.
	folds1&2: function fold lists, corresponding to terms in funcs1&2.
	sharenum: number of parameters shared by funcs1&2, should include both the fixed and non-fixed parameters.
	fix_index: int, slice or sequence of ints,
		    Object that defines the index or indices before which values is inserted. Check numpy.insert for more info.
	fix_params: array_like,
		    Values to insert into p0, represent the fixed parameters.
	p0: initial parameters guess, only include the non-fixed parameters. The fix_params and p0 combined based on fix_index will be used as the full parameter space in the model defined by funcs1, folds1, funcs2, folds2.
	frange: frequency range (low,high) bounds. low/high can be a list or a single items.
	bounds: parameters bounds, check scipy.optimize.curve_fit input.
	pltflag: if non-zero, will plot fitted curves for comparison.
	mainsize: int, the number of parameters representing the actual signal, the rest are viewed as backgrounds; if none, set mainsize=sharenum.
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
        if fitmode == 'g': #determine if gain correct fit is required
                y1=data.gx
                y2=data.gy
        elif fitmode == 'n':
                y1=data.nx
                y2=data.ny
        elif fitmode == 'gn':
                y1=data.gnx
                y2=data.gny
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
        model=assembleShare(model1,model2,sharenum) # sharenum should include the shared fixed parameters
        fitmodel=fixParam(model, fix_index, fix_param, p0)
        
        popt,pcov=scipy.optimize.curve_fit(fitmodel, x,y,p0=p0,bounds=bounds)
        perr=np.sqrt(np.diag(pcov))
        res=fitmodel(x,*popt)-y # calculate residual before update popt
        
        popt_full=np.insert(popt, fix_index, fix_param)
        _,popt1,popt2=paramUnfold(popt_full,funcs1,folds1,funcs2,folds2,sharenum)
        
        if pltflag:
                fig,axes=plt.subplots(2,2,figsize=figsize)
                fig.subplots_adjust(wspace=wspace,hspace=hspace)
                lines=Plotting.fitCheck_1sim(axes,data,fitmode,funcs1,funcs2,sharenum,popt1,popt2,res,mainsize=mainsize,frange=frange,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
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
def curve_linfit(x, y, reverse=False, roll_length=20, per_tol=1):
	'''
	2021-12-14 21:03
	Do a linear fit to the head or tail of a 2d curve. The program starts by linear fitting the first roll_length of points at the head/tail of the curve, and record the std deviation. Then increase roll_length while keep linear fitting the data. The program stops when the std deviation has changed more than the given percentage tolerance.
	Syntax:
	-------
	para, roll_length = curve_linfit(x, y[, reverse=False, roll_length=20, per_tol=1])
	Parameters:
	-----------
	x,y: lists or np.arrays, curve data.
	reverse: Boolean, sort x and y according to reverse, False is ascending x, True is descending x.
	roll_length: float, rolling length, the number of points for the initial linear fit.
	per_tol: float, percentage tolerance threshold before another linear fit is called impossible.
	Returns:
	--------
	para: list, linear fit parameters [slope, intercept].
	roll_length: int, length of the used data for the final fit.
	'''
	x_sorted = sorted(x, reverse=reverse)
	y_sorted = [ys for _,ys in sorted(zip(x, y), reverse=reverse)]

	#-----------------
	x_start = x_sorted[0:roll_length:] # data to start with
	y_start = y_sorted[0:roll_length:]

	para = np.polyfit(x_start, y_start, 1)
	residual = y_start - np.polyval(para, x_start)
	stddev = np.std(residual)
	tolerance = stddev * (1 + per_tol) # upper limit of stddev

	i=0
	while ((stddev<tolerance)&(roll_length+i+1<=len(x_sorted) )) :
	    i+=1
	    x_select = x_sorted[0:roll_length+i:]
	    y_select = y_sorted[0:roll_length+i:]
	    para = np.polyfit(x_select,y_select,1)
	    # calculate rolling residual
	    x_last = x_sorted[i:roll_length+i:] # crop the last roll_length points from included data
	    y_last = y_sorted[i:roll_length+i:]
	    residual = y_last - np.polyval(para, x_last)
	    stddev = np.std(residual)
	
	return para, roll_length+i
#=======================================================================
def curve_linfit_fixSlope(x, y, slope, reverse=False, roll_length=20, per_tol=1):
	'''
	2023-07-14 17:49
	Do a linear fit to the head or tail of a 2d curve while keeping the slope of the linear function constant in order to obtain the intercept. The program starts by linear fitting the first roll_length of points at the head/tail of the curve, and record the std deviation. Then increase roll_length while keep linear fitting the data. The program stops when the std deviation has changed more than the given percentage tolerance.
	Syntax:
	-------
	intercept, roll_length = curve_linfit_fixSlope(x, y, slope[, reverse=False, roll_length=20, per_tol=1])
	Parameters:
	-----------
	x,y: lists or np.arrays, curve data.
	slope: float, the slope of the linear function.
	reverse: Boolean, sort x and y according to reverse, False is ascending x, True is descending x.
	roll_length: float, rolling length, the number of points for the initial linear fit.
	per_tol: float, percentage tolerance threshold before another linear fit is called impossible.
	Returns:
	--------
	intercept: float, linear fitted intercept value.
	roll_length: int, length of the used data for the final fit.
	'''
	x_sorted = sorted(x, reverse=reverse)
	y_sorted = [ys for _,ys in sorted(zip(x, y), reverse=reverse)]
	x_sorted, y_sorted = np.asarray(x_sorted), np.asarray(y_sorted)

	#-----------------
	x_start = x_sorted[0:roll_length:] # data to start with
	y_start = y_sorted[0:roll_length:]

	intercept = np.mean(y_start)-slope*np.mean(x_start) # this is mathematically rigorous, you can prove it by minimizing the stddev
	residual = y_start - slope*x_start - intercept
	stddev = np.std(residual)
	tolerance = stddev * (1 + per_tol) # upper limit of stddev

	i=0
	while ((stddev<tolerance)&(roll_length+i+1<=len(x_sorted) )) :
	    i+=1
	    x_select = x_sorted[0:roll_length+i:]
	    y_select = y_sorted[0:roll_length+i:]
	    residual = y_select - slope*x_select - intercept

	    # calculate rolling residual
	    x_last = x_sorted[i:roll_length+i:] # crop the last roll_length points from included data
	    y_last = y_sorted[i:roll_length+i:]
	    residual = y_last - slope*x_last - intercept
	    stddev = np.std(residual)
	
	return intercept, roll_length+i
#=======================================================================


