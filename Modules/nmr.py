'''
NMR sweep signal data. There is no recorded time-domain. Time-domain is generated after data reading.
'''  
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import pandas as pd
import cmath
import ntpath

import FuncLib
import Functions as func
import Plotting
import readLog
from readLog import nmrLog
from readLog import sweepLog as swplog


#=======================================================================
class nmr_old(object):
	'''
	DEPRECATED
	NMR FID signal class.
	The read file doesn't contain time info. Time is associated with data during initialization.
	Syntax:
	-------
	data=nmr(filename[,tstep=2e-7,zerofillnum=0,logname='NLog_001.dat'])
	Parameters:
	-----------
	filename: nmr file name.
	tstep: the associated time is assumed with a step spacing = tstep.
	zerofillnum: extra number of points appended during zerofilling.
	logname: log file path.
	Returns:
	--------
	self.filename: filename.
	self.logname: logname.
	self.log: associated row in the log.
	self.date: str, date of sweep.
	self.time: str, time of sweep.
	self.datetime: datetime.datetime class time format.
	self.epoch: epoch seconds from UTC 1970-1-1_00:00:00.
	self.Cmct: mct capacitance in pF.
	self.Tmct: mct temperature from labview record.
	self.plm: plm display.
	self.nmrabs: nmr absolute sum.
	self.nmrfilter: nmr filtered absolute sum from labview record.

	self.nmr: nmr FID data array.
	self.numpts: number of points in raw data.
	self.t: time array associated with raw data.
	self.fftnmr: FFT of self.nmr.
	self.f: frequency array associated with self.fftnmr.
	self.nmr0fill: zero-filled self.nmr.
	self.numpts0fill: total number of points after zerofilling.
	self.t0fill: time array associated with zero-filled nmr FID data.
	self.fftnmr0fill: FFT of self.nmr0fill.
	self.fftnmr0fill_m: magnitude of self.fftnmr0fill
	self.fftnmr0fill_ph: phase of self.fftnmr0fill
	self.f0fill: frequency array associated with self.fftnmr0fill.

	--After running self.smooth(..):
	self.sfftnmr0fill: smoothed self.fftnmr0fill.
	self.popt: fitted parameters of peaks in FFT FID spectrum.
	'''
	def __init__(self,filename,tstep=2e-7,zerofillnum=0,logname='NLog_001.dat'):
		self.filename=filename
		self.zerofillnum=zerofillnum
		
		f0=open(filename)
		self.nmr=np.loadtxt(f0)
		self.numpts=len(self.nmr)
		self.t=np.linspace(0,tstep*(self.numpts-1),self.numpts)
		self.fftnmr=fftpack.fft(self.nmr)
		self.f=FuncLib.domainfft(self.t)

		self.nmr0fill=np.append(self.nmr,np.zeros(zerofillnum))
		self.numpts0fill=self.numpts+zerofillnum
		self.t0fill=np.linspace(0,tstep*(self.numpts0fill-1),self.numpts0fill)
		self.fftnmr0fill=fftpack.fft(self.nmr0fill)
		self.f0fill=FuncLib.domainfft(self.t0fill)
#-----------------------------------------------------------------------
# the log file row which contains the info associated with this nmr file
		self.logname=logname
		nl=nmrLog(logname)
		matchCond=nl.nmrfilename==filename #filename match condtion
		#fetching info about the matched row
		self.log=nl.log[matchCond]
		self.date=nl.date[matchCond].item()
		self.time=nl.time[matchCond].item()
		self.datetime=nl.datetime[matchCond].item()
		self.epoch=nl.epoch[matchCond].item()
		self.Cmct=nl.Cmct[matchCond].item()
		self.Tmct=nl.Tmct[matchCond].item()
		self.plm=nl.plm[matchCond].item()
		self.nmrabs=nl.nmrabs[matchCond].item()
		self.nmrfilter=nl.nmrfilter[matchCond].item()
#=======================================================================
	@property
	def fftnmr0fill_m(self):
		'''
		Magnitude list of the fftnmr0fill complex number list.
		'''
		vfunc=np.vectorize(cmath.polar)
		m,_=vfunc(self.fftnmr0fill)
		return m
#-----------------------------------------------------------------------
	@property
	def fftnmr0fill_ph(self):
		'''
		Phase list of the fftnmr0fill complex number list. Unit is rad.
		'''
		vfunc=np.vectorize(cmath.polar)
		_,phase=vfunc(self.fftnmr0fill)
		return phase
#=======================================================================
	def smooth(self,frange,window_size,order,deriv=0,rate=1):
		'''
		Smooth fftnmr0fill with a Savitzky-Golay filter.
		Syntax:
		-------
		sfftnmr0fill=smooth(frange,window_size,order[,deriv=0,rate=1])
		frange: Frequency range array, [low,high] format. Signal within range will be smoothed, boundary included.
		window_size: int; the length of the window. Must be odd integer.
		order: int; order of polynomial in fitting.
		deriv: int; order of derivative to compute.
		rate: inherited parameter from savitzky_golay method.
		Returns:
		--------
		sfftnmr0fill: smoothed FFT of zero-filled nmr FID.
		Note:
		-----
		Check Functions.savitzky_golay for details.
		'''
		condition=(self.f0fill>=frange[0])&(self.f0fill<=frange[1])
		conditionlow=(self.f0fill<frange[0])
		conditionhigh=(self.f0fill>frange[1])
		yr=self.fftnmr0fill.real[condition]
		yi=self.fftnmr0fill.imag[condition]
		smoothyr=func.savitzky_golay(yr,window_size,order,deriv=deriv,rate=rate)
		smoothyi=func.savitzky_golay(yi,window_size,order,deriv=deriv,rate=rate)
		smoothfftnmr0fillr=np.concatenate((self.fftnmr0fill.real[conditionlow], smoothyr, self.fftnmr0fill.real[conditionhigh]))
		smoothfftnmr0filli=np.concatenate((self.fftnmr0fill.imag[conditionlow], smoothyi, self.fftnmr0fill.imag[conditionhigh]))
		sfftnmr0fill=smoothfftnmr0fillr+1j*smoothfftnmr0filli
		setattr(self,'sfftnmr0fill',sfftnmr0fill)
		return sfftnmr0fill
#=======================================================================
	def plot(self,pltmode,figsize=(15,5),wspace=0.4,hspace=0.3,iter=0,fillstyle='full',markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
		'''
		Plot nmr signal.
		Syntax:
		plot(pltmode[,figsize=(15,5),wspace=0.4,hspace=0.3,iter=0,fillstyle='full',markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
		pltmode: str, plot mode, examples: 'td'=time_vs_FID, 'fi'=frequency_vs_fftFIDimag... Can recognize 't/d/f/r/i/m/p' meaning 'time/free-induction-decay/frequency/fft-real-part/fft-imagenary-part/fft-magnitude/phase' respectively, pltmode is case insensitive.
		figsize,wspace,hspace,iter,fillstyle,markeredgewidth,markersize,linewidth,legloc,bbox_to_anchor,legsize: plot settings.
		Returns:
		fig,ax,line: respective plotting handles.
		'''
		pltmode=pltmode.lower()
		if 'all' not in pltmode: #individual plot
			fig,axis=plt.subplots(1,1,figsize=figsize)
			line=Plotting.nmrSingle(axis,self,pltmode,iter=iter,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axis,line
		else: #1x3 subplots
			fig,axes=plt.subplots(1,3,figsize=figsize)
			fig.subplots_adjust(wspace=wspace,hspace=hspace)
			line=Plotting.nmrAll(axes,self,iter=iter,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axes,line
#=======================================================================
	def fit(self,p0,window_size,order,sfrange=(-np.inf,np.inf),deriv=0,rate=1,bounds=(-np.inf,np.inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8):
		'''
		Fit self.sfftnmr0fill to several peaks, each peak described by 4 parameters. Smoothing procedure is included in this process.
		Syntax:
		-------
		popt,pcov,perr[,fig,axes,lines]=fit(p0,window_size,order[,sfrange=(-inf,inf),deriv=0,rate=1,bounds=(-inf,inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8])
		Parameters:
		-----------
		p0: Initial fitting parameters, format is [s01,T1,f01,phase1,s02,T2,f02,phase2,...], length=4xN.
		window_size,order,deriv,rate: func.savitzky_golay smooth method input parameters.
		sfrange: Frequency range to perform smoothing, boundary included.
		bounds: Optimization parameter bounds for scipy.optimize.curve_fit.
		pltflag: plot flag.
		figsize,wspace,hspace: figure and subplots spacing settings.
		marker,markersize,linewidth: Fitted curve plot settings.
		legloc,bbox_to_anchor,legsize: legend position and size.
		Returns:
		--------
		popt: Fitting parameters optimized from p0.
		pcov: Covariance output from scipy.optimize.curve_fit.
		perr: Standard deviation associated with popt.
		fig: figure handle.
		axes: 1x3 axes handles array.
		lines: lines output from Plotting.fitChecknmr().
		Note:
		-----
		Create attribute self.popt after running.
		fig/axes/lines: Only output when pltflag=1.
		'''
		if pltflag:
			popt,pcov,perr,fig,axes,lines=func.nmr1simfit(self,p0,window_size,order,sfrange=sfrange,deriv=deriv,rate=rate,bounds=bounds,pltflag=pltflag,figsize=figsize,wspace=wspace,hspace=hspace,marker=marker,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
		else:
			popt,pcov,perr=func.nmr1simfit(self,p0,window_size,order,sfrange=sfrange,deriv=deriv,rate=rate,bounds=bounds,pltflag=pltflag)

		setattr(self,'popt',popt)

		if pltflag:
			return popt,pcov,perr,fig,axes,lines
		return popt,pcov,perr
#=======================================================================
class nmr(object):
	'''
	2019-09-25 16:18
	NMR FID signal class.
	The read file doesn't contain time info. Time is stored in the log file.
	Syntax:
	-------
	data=nmr(path[,zerofillnum=0,logpath=None,dtLabel='dt_s',dt=2e-7])
	Parameters:
	-----------
	path: str; nmr file path.
	zerofillnum: int; extra number of points appended during zerofilling.
	logpath: str; log file path
	dtLabel: str; the attribute that should contain the time step info.
	dt: float; time step of the FID. This input is only used if not specified in the log.
	Returns:
	--------
	self._path: str; nmr file path.
	self._filename: str; nmr file basename.
	self._zerofillnum: int; zerofillnum.
	self._content: pandas.DataFrame; entire sweep content.
	self.item: pandas.Series; each item is one column with the same name (but lower case) in the content.
	self._logpath: str; log file path.
	self._logname: str; log file basename.
	self._log: pandas.DataFrame; row of log content corresponding to this sweep file.
	self._dtLabel: str; same as input dtLabel. 
	self.logitem: each logitem is one item with the same name in the self._log.
	self._datetime: datetime; date time of the nmr sweep.
	self._epoch: float; epoch time of the sweep.
	self._dt: float; time step of the FID. It is specified by dtLabel in the log, if log doesn't contain info as specified by dtLabel, the input dt will be used.
	self._t: numpy.array; time array associated with voltage data.
	self._t0: numpy.array; time array with zero filling.
	self._f: numpy.array; fft frequency generated from self._t.
	self._f0fill: numpy.array; fft frequency generated from self._t0fill.
	self._nmr: numpy.array; nmr FID data array. same as self.voltagev.
	self._nmr0fill: numpy.array; zero-filled self._nmr.
	self._fftnmr: numpy.array; FFT of self._nmr.
	self._fftnmr0fill: numpy.array; FFT of self._nmr0fill.
	self._fftnmr0fill_m: numpy.array; magnitude of self._fftnmr0fill.
	self._fftnmr0fill_ph: numpy.array; phase of self._fftnmr0fill.
	'''
	def __init__(self,path,zerofillnum=0,logpath=None,dtLabel='dt_s',dt=2e-7):
		self._path=path
		self._filename=ntpath.basename(path)
		self._zerofillnum=zerofillnum
		self._content=pd.read_csv(path,names=['VoltageV'])
		self._dtLabel=dtLabel

		# assign pandas.Series to attributes based on name
		col_names=self._content.columns.tolist()
		for name in col_names:
			setattr(self,name.lower(),self._content[name].values)
#-----------------------------------------------------------------------
		# read log content row
		if logpath is not None:
			self._logpath=logpath
			self._logname=ntpath.basename(logpath)
		# the log file row which contains the info associated with this sweep
			swpl=swplog(logpath)
			matchCond=swpl.filename==self._filename #search for the row representing row of designated filename, this is assumed to be contained in the attribute 'filename' of swpl.
			self._log=swpl._content[matchCond] # get specific log row
			if not self._log.empty: # if log match is found, load log info to individual attributes
				for name in self._log.columns:
					# assign self._log items to attributes
					setattr(self,name.lower(),self._log[name].item())
				if hasattr(swpl,'_datetime'): # only load datetime & epoch when the log contains these info.
					# assign self._datetime and self._epoch
					self._datetime=swpl._datetime[matchCond].item()
					self._epoch=swpl._epoch[matchCond].item()
#-----------------------------------------------------------------------
		# get time step from log or input
		if hasattr(self,dtLabel): # if log has required info
			self._dt=getattr(self,self._dtLabel)
		else: # read from input if log is missing info
			self._dt=dt
		self._nmr=self.voltagev # only here due to naming habit 
		self._fftnmr=fftpack.fft(self._nmr)
		self._nmr0fill=np.append(self._nmr,np.zeros(zerofillnum))
		self._fftnmr0fill=fftpack.fft(self._nmr0fill)
#=======================================================================
	@property
	def _t(self):
		'''
		time array of the sweep.
		'''
		return np.linspace(0,self._dt*(self._nmr.size-1),self._content.size)
#-----------------------------------------------------------------------
	@property
	def _t0fill(self):
		'''
		time array with zero filling.
		'''
		return np.linspace(0,self._dt*(self._nmr0fill.size-1),self._nmr0fill.size)
#-----------------------------------------------------------------------
	@property
	def _f(self):
		'''
		fft frequency domain.
		'''
		return FuncLib.domainfft(self._t)
#-----------------------------------------------------------------------
	@property
	def _f0fill(self):
		'''
		fft frequency domain with zero filling.
		'''
		return FuncLib.domainfft(self._t0fill)
#-----------------------------------------------------------------------
	@property
	def _fftnmr0fill_m(self):
		'''
		Magnitude list of the fftnmr0fill complex number list.
		'''
		vfunc=np.vectorize(cmath.polar)
		m,_=vfunc(self._fftnmr0fill)
		return m
#-----------------------------------------------------------------------
	@property
	def _fftnmr0fill_ph(self):
		'''
		Phase list of the fftnmrf0fill complex values. Unit is rad.
		'''
		vfunc=np.vectorize(cmath.polar)
		_,phase=vfunc(self._fftnmr0fill)
		return phase
#=======================================================================
	def plot(self,pltmode,figsize=(15,5),wspace=0.4,hspace=0.3,iter=0,fillstyle='full',markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
		'''
		2019-09-25 16:18
		Plot nmr signal.
		Syntax:
		plot(pltmode[,figsize=(15,5),wspace=0.4,hspace=0.3,iter=0,fillstyle='full',markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
		pltmode: str, plot mode, examples: 'td'=time_vs_FID, 'fi'=frequency_vs_fftFIDimag... Can recognize 't/d/f/r/i/m/p' meaning 'time/free-induction-decay/frequency/fft-real-part/fft-imagenary-part/fft-magnitude/phase' respectively, pltmode is case insensitive.
		figsize,wspace,hspace,iter,fillstyle,markeredgewidth,markersize,linewidth,legloc,bbox_to_anchor,legsize: plot settings.
		Returns:
		fig,ax,line: respective plotting handles.
		'''
		pltmode=pltmode.lower()
		if 'all' not in pltmode: #individual plot
			fig,axis=plt.subplots(1,1,figsize=figsize)
			line=Plotting.nmr_single(axis,self,pltmode,iter=iter,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axis,line
		else: #1x3 subplots
			fig,axes=plt.subplots(1,3,figsize=figsize)
			fig.subplots_adjust(wspace=wspace,hspace=hspace)
			line=Plotting.nmr_all(axes,self,iter=iter,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axes,line
#=======================================================================
	def fit(self,p0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8):
		'''
		2019-09-25 17:34
		Fit self._fftnmr0fill to several peaks, each peak described by 4 parameters.	
		Syntax:
		-------
		popt,pcov,perr[,fig,axes,lines]=fit(p0[,bounds=(-inf,inf),pltflag=0,figsize=(16,5),wspace=0.4,hspace=0.2,marker='.',markersize=1,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=8])
		Parameters:
		-----------
		p0: Initial fitting parameters, format is [s01,T1,f01,phase1,s02,T2,f02,phase2,...], length=4xN.
		frange: (lb,ub); lower and upper bound of the frequency range for fitting.
		bounds: Optimization parameter bounds for scipy.optimize.curve_fit.
		pltflag: plot flag.
		figsize,wspace,hspace: figure and subplots spacing settings.
		marker,markersize,linewidth: Fitted curve plot settings.
		legloc,bbox_to_anchor,legsize: legend position and size.
		Returns:
		--------
		popt: Fitting parameters optimized from p0.
		pcov: Covariance output from scipy.optimize.curve_fit.
		perr: Standard deviation associated with popt.
		fig: figure handle.
		axes: 1x3 axes handles array.
		lines: lines output from Plotting.fitChecknmr().
		Note:
		-----
		Create attribute self.popt after running.
		fig/axes/lines: Only output when pltflag=1.
		'''
		if pltflag:
			popt,pcov,perr,fig,axes,lines=func.nmr_1simfit(self,p0,frange=frange,bounds=bounds,pltflag=pltflag,figsize=figsize,wspace=wspace,hspace=hspace,marker=marker,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
		else:
			popt,pcov,perr=func.nmr_1simfit(self,p0,frange=frange,bounds=bounds,pltflag=pltflag)

		setattr(self,'popt',popt)

		if pltflag:
			return popt,pcov,perr,fig,axes,lines
		return popt,pcov,perr
#=======================================================================

