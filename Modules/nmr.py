'''
NMR sweep signal data. There is no recorded time-domain. Time-domain is generated after data reading.
'''  
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import pandas as pd
import cmath

import FuncLib
import Functions as func
import Plotting
import readLog
from readLog import nmrLog

#=======================================================================
class nmr(object):
	'''
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
		lines: lines output from Plotting.fitChecknmr()a.
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
