'''
Single frequency sweep object class.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import warnings

import Functions as func
import Plotting
import Utility as utl
import readLog
from readLog import freqSweepLog as fswplog

#=======================================================================
class FreqSweep(object):
	'''
	Class object of frequency sweeps.
	Syntax:
	-------
	self=FreqSweep(filename,logname=None[,ftimes=1,xtimes=1,ytimes=1,rtimes=1,correctFunc=utl.gainCorrect])
	Parameters:
	-----------
	filename: name = device + filenum.
	f/x/y/rtimes: multiply data.f/x/y/r by this during initialization.
	correctFunc: rolloff gain correction function.
	logname: log_file_name in which this file's meta data is recorded.
	Returns:
	--------
	self.filename: filename.
	self.logname: logname, only exists when there's input.

	self.f/.x/.y/.r/.Tmc/.Tmct/.Cmct: f,x,y,r,Tmc,Tmct,Cmct.
	self.gx/.gy/.gr: gain-corrected x/y/r, only created when called.
	self.nx/.ny/.nr: excitation-normalized x/y/r.
	self.gnx/.gny/.gnr: gain-corrected and excitation-normalized x/y/r.
	self.avgTmc/.avgTmct: average Tmc and Tmct
	self.Tmm: choose between average Tmc and Tmct (check Utility.chooseT for criteria)
	self.gcorrect: used correctFunc.
	
	self.log: the log file row associated with this sweep.
	self.date: str, sweep date.
	self.time: str, sweep start time.
	self.datetime: datetime.datetime class time format.
	self.epoch: epoch seconds from UTC 1970-01-01_00:00:00.
	self.batchnum: sweep batch number.
	self.vl: low frequency excitation in Vpp.
	self.nmrfilename: associated NMR file name.
	self.plm: PLM display.
	self.nmrfilter: NMR file filtered absolute sum.

	--After running self.lrtz1simfit(..):
	self.popt: optimized fitting paramters from the latest fit.
	self.popt1&2: optimized fitting parameters from simultaneous fitting.
	Notes:
	------
	logname is not an optional input.
	'''
	def __init__(self,filename,ftimes=1,xtimes=1,ytimes=1,rtimes=1,correctFunc=utl.gainCorrect,logname=None):
		self.filename=filename
		self.f, self.x, self.y, self.r, self.Tmc, self.Tmct, self.Cmct=utl.sweepLoad(filename,ftimes,xtimes,ytimes,rtimes) #load self.f/x/y/r/Tmc/Tmct from filename to self
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			self.avgTmc,self.avgTmct,self.avgCmct=np.nanmean((self.Tmc,self.Tmct,self.Cmct),axis=1)#suppress np.nanmean warnings for all np.nan arrays.
		self.Tmm=utl.chooseT(self.avgTmc,self.avgTmct)
		self.gcorrect=correctFunc
#-----------------------------------------------------------------------
# the log file row which contains the info associated with this sweep
		self.logname=logname
		fswpl=fswplog(logname)
		matchCond=fswpl.fswpfilename==filename #filename match condition
		#fetching info about the matched row
		self.log=fswpl.log[matchCond]
		self.date=fswpl.date[matchCond].item()
		self.time=fswpl.time[matchCond].item()
		self.datetime=fswpl.datetime[matchCond].item()
		self.epoch=fswpl.epoch[matchCond].item()
		self.batchnum=fswpl.batchnum[matchCond].item()
		self.frange=fswpl.frange[matchCond].item()
		self.flow=fswpl.flow[matchCond].item()
		self.fhigh=fswpl.fhigh[matchCond].item()
		self.swptime=fswpl.swptime[matchCond].item()
		self.numpts=fswpl.numpts[matchCond].item()
		self.LIsens=fswpl.LIsens[matchCond].item()
		self.LITconst=fswpl.LITconst[matchCond].item()
		self.LIphase=fswpl.LIphase[matchCond].item()
		self.updown=fswpl.updown[matchCond].item()
		self.vl=fswpl.vl[matchCond].item()
		self.nmrfilename=fswpl.nmrfilename[matchCond].item()
		self.plm=fswpl.plm[matchCond].item()
		self.nmrfilter=fswpl.nmrfilter[matchCond].item()

#=======================================================================
	@property
	def gx(self):
		'''
		Create self.gx attribute containing rolloff corrected signal.
		'''
		return self.gcorrect(self.f,self.x)
#-----------------------------------------------------------------------
	@property
	def gy(self):
		'''
		Create self.gy attribute containing rolloff corrected signal.
		'''
		return self.gcorrect(self.f,self.y)
#-----------------------------------------------------------------------
	@property
	def gr(self):
		'''
		Create self.gr attribute containing rolloff corrected signal.
		'''
		return self.gcorrect(self.f,self.r)
#-----------------------------------------------------------------------
	@property
	def nx(self):
		'''
		Normalized x-channel to excitation.
		'''
		return self.x/self.vl
#-----------------------------------------------------------------------
	@property
	def ny(self):
		'''
		Normalized y-channel to excitation.
		'''
		return self.y/self.vl
#-----------------------------------------------------------------------
	@property
	def nr(self):
		'''
		Normalized r-channel to excitation.
		'''
		return self.r/self.vl
#-----------------------------------------------------------------------
	@property
	def gnx(self):
		'''
		Gain corrected and normalized x-channel.
		'''
		return self.gx/self.vl
#-----------------------------------------------------------------------
	@property
	def gny(self):
		'''
		Gain corrected and normalized y-channel.
		'''
		return self.gy/self.vl
#-----------------------------------------------------------------------
	@property
	def gnr(self):
		'''
		Gain corrected and normalized r-channel.
		'''
		return self.gr/self.vl
#=======================================================================
	def mctC2T(self,p,branch='low',Pn=34.3934):
		'''
		Calculate temperature, T, from capacitance, C.
		Syntax:
		-------
		T=mctC2T(p[,branch='low',Pn=34.3934])
		Parameters:
		-----------
		p: prefactor parameters for utl.mctC2P. P=p0+p1*C+p2*C^2+...+pn*C^n.
		branch: the melting curve branch in which the solution is searched for. 'low' is below 315.23959351mK, which is the lowest pressure point on the melting curve, 'high' is above this point.
		Pn: Neel transition pressure in bar.
		Returns:
		--------
		T: temperature in mK.
		Notes:
		------
		The returned T may contain np.nan values.
		'''
		C=self.Cmct
		P=utl.mctC2P(C,p)
		T=utl.mctP2T(P,branch=branch,Pn=Pn)

		# set Pmct attributes, and update Tmct related attributes
		setattr(self,'Pmct',P)
		setattr(self,'avgPmct',np.nanmean(P))
		setattr(self,'Tmct',T)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			setattr(self,'avgTmct',np.nanmean(T))#ignore nanmean warnings, if any
		setattr(self,'Tmm',utl.chooseT(self.avgTmc,self.avgTmct))
		return T
#=======================================================================
	def plot(self,pltmode,figsize=(12,9),wspace=0.4,hspace=0.3,iter=0,fillstyle='full',markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
		'''
		Plot by having x- and y-axes assigned based on pltmode.
		rotates through 12 colors, 23 markers, 4 linestyles	
		Syntax:
		-------
		fig,axes,line=self.plot(pltmode[,figsize=(12,9),wspace=0.4,hspace=0.3,iter=0,fillstyle='full',markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
		Parameters:
		-----------
		pltmode: example: 'fx'=>f vs x; 'xy'=>x vs y; 'rf'=>r vs f; 'yy'=>y vs y; only f,x,y,r are acceptable options.
			pltmode=='all' will plot in a 'fx'+'fy'+'fr'+'xy' fashion
			can plot .gx/.gy/.gr if add 'g' to pltmode, position/case insensitive, e.g. 'fgx'=='fxG'=='gFx','ALlg'=='gall', etc.
			can plot .nx/.ny/.nr if add 'n' to pltmode, or plot .gnx/.gny/.gnr if both 'g' and 'n' in pltmode.
		iter,fillstyle,markeredgewidth,markersize,linewidth,figsize,wspace,hspace: axes and fig settings.
		legloc: legend location.
		bbox_to_anchor: legend anchor point.
		legsize: legend font size.
		Returns:
		--------
		fig,axes,line: handles, line is a tuple of all line instances.
		'''
		pltmode=pltmode.lower()
		if 'all' not in pltmode: #individual plot with input requirement
			fig,axis=plt.subplots(1,1,figsize=figsize)
			line=Plotting.sweepSingle(axis,self,pltmode,iter=iter,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axis,line #returns fig,axie,line handles for future use
		else: #2*2 subplots of 'fx','fy','fr','xy'
			fig,axes=plt.subplots(2,2,figsize=figsize)
			fig.subplots_adjust(wspace=wspace,hspace=hspace)
			line=Plotting.sweepAll(axes,self,pltmode,iter=iter,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axes,line
#=======================================================================
	def lrtz1simfit(self,fitmode,funcs1,funcs2,sharenum,p0,folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
		'''
		Simultaneously fit x-&y-channels, with x in front. Plot fitted curve if demanded.
		Syntax:
		-------
		popt,pcov,perr,res,popt1,popt2[,fig,axes,lines]=lrtz1simfit(fitmode,funcs1,funcs2,sharenum,p0[,folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor(0,1),legsize=10])
		Parameters:
		-----------
		fitmode: only difference is if it contains 'g' or not, 'g' will introduce rolloff gain correct.
		funcs1&2: function lists of models for simultaneous fitting.	
		sharenum: number of parameters shared by funcs1&2.
		p0: initial parameters guess.
		folds1&2: function fold lists, corresponding to terms in funcs1&2.
		frange: frequency range (low,high) bounds.
		bounds: parameters bounds, check scipy.optimize.curve_fit input.
		pltflag: if non-zero, will plot fitted curves for comparison.
		figsize: figure size.
		wspace,hspace: width/horizontal spacing between subplots.
		markersize,linewidth: matplotlib.pyplot.plot inputs.
		legloc: legend location.
		bbox_to_anchor: legend anchor point.
		legsize: legend font size.
		Default for optional inputs:
			folds1&2=np.ones(len(funcs1&2)),frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pltflag=0,figsize=(12,9),wspace=0.4,hspace=0.3,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10.
		Returns:
		--------
		popt: fitted parameters, folds1&2 influence removed.
		pcov: 2d array, the estimated covariance of popt.
		perr: standard deivation associated with popt.
		res: residual = calculated values from popt - data values.
		popt1&2: popt separated into two parts corresponding to funcs1&2.
		fig: figure handle.
		axes: 2x2 axes handles array.
		lines: lines output from Plotting.fitCheckSim().
		Note:
		-----
		Sets attributes self.popt/.popt1/.popt2 to popt/popt1/popt2, create these attributes if previously nonexistent.
		fig/axes/lines: Only output when pltflag=1.
		'''
		if folds1 is None: # assign folds1&2's default values as ones.
			folds1=np.ones(len(funcs1))
		if folds2 is None:
			folds2=np.ones(len(funcs2))
		
		if pltflag:
			popt,pcov,perr,res,popt1,popt2,fig,axes,lines=func.lrtz1simfit(self,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0,frange=frange,bounds=bounds,pltflag=pltflag,figsize=figsize,wspace=wspace,hspace=hspace,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
		else:
			popt,pcov,perr,res,popt1,popt2=func.lrtz1simfit(self,fitmode,funcs1,folds1,funcs2,folds2,sharenum,p0,frange=frange,bounds=bounds,pltflag=pltflag)

		setattr(self,'popt',popt) # set popt,popt1&2 as attributes
		setattr(self,'popt1',popt1)
		setattr(self,'popt2',popt2)

		if pltflag:
			return popt,pcov,perr,res,popt1,popt2,fig,axes,lines
		return popt,pcov,perr,res,popt1,popt2
#=======================================================================
