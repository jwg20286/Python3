'''
sweep.py: Ver 1.0.
Sweep object classes.
'''
import pandas as pd
import matplotlib.pyplot as plt

import readLog
from readLog import sweepLog as swplog
import Utility as utl
import Plotting
#=======================================================================
class singleSweep(object):
	'''
	Class of a single sweep.
	The sweep column headers are converted to lower cases when assigned to attributes, without underscore in front. Other attributes start with one underscore.
	This class assumes that its instance and the log files have different headers for all their data columns.
	Syntax:
	-------
	self=singleSweep(filename,[logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp'])
	Parameters:
	-----------
	filename: str, filename of the loaded sweep file.
	logname: str, log_file_name in which this file's metadata is stored.
	correctFunc: function, gain correcting function accounting for frequency rolloff of the lock in, etc.; used when 'g(n)x/y/r' are called.
	normByParam: str, when '(g)nx/y/r' are called, they will be divided ("normalized") by this named attribute of the instance.
	Returns:
	--------
	self._filename: str, loaded filename.
	self._content: pandas.DataFrame, entire sweep content.
	self._gcorrect: gain correct function.
	self.item: pandas.Series, each item is one column with the same name (but lower case) in the content.
	self._logname: log file name.
	self._log: row of log content corresponding to this sweep file.
	self.logitem: each logitem is one item with the same name in the self._log.
	self._datetime: datetime.datetime, datetime of the loaded sweep file.
	self._epoch: float, epoch seconds of the loaded sweep file.

	--when called:
	self.(g)(n)x/y/r: pandas.Series, if x,y,r exist, they can be gain-corrected with the given correctFunc to account for lockin rolloff, etc.; they can be normalized by the given attribute specified by normByParam; 'gn' can appear together meaning both methods are implemented.
        '''
	def __init__(self,filename,logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp'):
		self._filename=filename
		self._content=pd.read_csv(filename,delim_whitespace=True)
		self._gcorrect=correctFunc
		self._normByParam=normByParam.lower()
#-----------------------------------------------------------------------
		# assign pandas.Series to attributes based on name
		col_names=self._content.columns.tolist()
		for name in col_names:
			setattr(self,name.lower(),self._content[name]) 
#-----------------------------------------------------------------------
		# the log file row which contains the info associated with this sweep
		if logname is not None:
			self._logname=logname
			swpl=swplog(logname)
			matchCond=swpl.filename==filename #search for the row representing row of designated filename, this is assumed to be contained in the attribute 'filename' of swpl.
			self._log=swpl._content[matchCond] # get specific log row
			if not self._log.empty: # if log match is found, load log info to individual attributes
				for name in self._log.columns:
					# assign self._log items to attributes
					setattr(self,name.lower(),self._log[name].item())
				if hasattr(swpl,'_datetime'): # only load datetime & epoch when the log contains these info.
					# assign self._datetime and self._epoch
					self._datetime=swpl._datetime[matchCond].item()
					self._epoch=swpl._epoch[matchCond].item()
#=======================================================================
	@property
	def gx(self):
		'''
		Create self.gx attribute containing rolloff corrected signal.
		'''
		return pd.Series(self._gcorrect(self.f,self.x))
#-----------------------------------------------------------------------
	@property
	def gy(self):
		'''
		Create self.gy attribute containing rolloff corrected signal.
		'''
		return pd.Series(self._gcorrect(self.f,self.y))
#-----------------------------------------------------------------------
	@property
	def gr(self):
		'''
		Create self.gr attribute containing rolloff corrected signal.
		'''
		return pd.Series(self._gcorrect(self.f,self.r))
#-----------------------------------------------------------------------
	@property
	def nx(self):
		'''
		Normalized x-channel to excitation.
		'''
		return pd.Series(self.x/getattr(self,self._normByParam))
#-----------------------------------------------------------------------
	@property
	def ny(self):
		'''
		Normalized y-channel to excitation.
		'''
		return pd.Series(self.y/getattr(self,self._normByParam))
#-----------------------------------------------------------------------
	@property
	def nr(self):
		'''
		Normalized r-channel to excitation.
		'''
		return pd.Series(self.r/getattr(self,self._normByParam))
#-----------------------------------------------------------------------
	@property
	def gnx(self):
		'''
		Gain corrected and normalized x-channel.
		'''
		return pd.Series(self.gx/getattr(self,self._normByParam))
#-----------------------------------------------------------------------
	@property
	def gny(self):
		'''
		Gain corrected and normalized y-channel.
		'''
		return pd.Series(self.gy/getattr(self,self._normByParam))
#-----------------------------------------------------------------------
	@property
	def gnr(self):
		'''
		Gain corrected and normalized r-channel.
		'''
		return pd.Series(self.gr/getattr(self,self._normByParam))
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
		C=self.cmct.values #numpy array
		P=utl.mctC2P(C,p)
		T=utl.mctP2T(P,branch=branch,Pn=Pn)

		# set Pmct attributes, and update Tmct related attributes
		# the saved format is pandas.Series
		setattr(self,'pmct',pd.Series(P))
		setattr(self,'tmct',pd.Series(T))
		return T
#=======================================================================
	def plot(self,pltmode,figsize=(12,7),wspace=0.4,hspace=0.3,fillstyle='full',iter_color=0,iter_marker=0,iter_linestyle=0,markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10):
		'''
		Plot by having x- and y-axes assigned based on pltmode.
		rotates through 12 colors, 23 markers, 4 linestyles	
		Syntax:
		-------
		fig,axes,line=self.plot(pltmode[,figsize=(12,7),wspace=0.4,hspace=0.3,fillstyle='full',iter_color=0,iter_marker=0,iter_linestyle=0,markeredgewidth=0.5,markersize=4,linewidth=1,legloc='lower left',bbox_to_anchor=(0,1),legsize=10])
		Parameters:
		-----------
		pltmode: example: 'fx'=>f vs x; 'xy'=>x vs y; 'rf'=>r vs f; 'yy'=>y vs y; only f,x,y,r are acceptable options.
			pltmode=='all' will plot in a 'fx'+'fy'+'fr'+'xy' fashion
			can plot .gx/.gy/.gr if add 'g' to pltmode, position/case insensitive, e.g. 'fgx'=='fxG'=='gFx','ALlg'=='gall', etc.
			can plot .nx/.ny/.nr if add 'n' to pltmode, or plot .gnx/.gny/.gnr if both 'g' and 'n' in pltmode.
		iter_color/marker/linestyle,fillstyle,markeredgewidth,markersize,linewidth,figsize,wspace,hspace: axes and fig settings.
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
			line=Plotting.sweep_single(axis,self,pltmode,iter_color=iter_color,iter_marker=iter_marker,iter_linestyle=iter_linestyle,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axis,line #returns fig,axie,line handles for future use
		else: #2*2 subplots of 'fx','fy','fr','xy'
			fig,axes=plt.subplots(2,2,figsize=figsize)
			fig.subplots_adjust(wspace=wspace,hspace=hspace)
			line=Plotting.sweep_all(axes,self,pltmode,iter_color=iter_color,iter_marker=iter_marker,iter_linestyle=iter_linestyle,fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linewidth=linewidth,legloc=legloc,bbox_to_anchor=bbox_to_anchor,legsize=legsize)
			return fig,axes,line
#=======================================================================

