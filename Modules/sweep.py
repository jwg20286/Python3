'''
sweep.py: Ver 1.0.
Sweep object classes.
'''
import pandas as pd

import readLog
from readLog import sweepLog as swplog
import Utility as utl
#=======================================================================
class singleSweep(object):
	'''
	Class of a single sweep.
	Syntax:
	-------
	self=singleSweep(filename,logname=None[,correctFunc=utl.gainCorrect)
	Parameters:
	-----------
	self._filename: str, loaded filename.
	self._content: pandas.DataFrame, entire sweep content.
	self._gcorrect: gain correct function.
	self.item: pandas.Series, each item is one column with the same name (but lower case) in the content.
	self._logname: log file name.
	self._log: row of log content corresponding to this sweep file.
	self.logitem: each logitem is one item with the same name in the self._log.
	self._datetime: datetime.datetime, datetime of the loaded sweep file.
	self._epoch: float, epoch seconds of the loaded sweep file.
        '''
	def __init__(self,filename,logname=None,correctFunc=utl.gainCorrect):
		self._filename=filename
		self._content=pd.read_csv(filename,delim_whitespace=True)
		self._gcorrect=correctFunc
		col_names=self._content.columns.tolist()
		for name in col_names:
			# assign pandas.Series to attributes based on name
			setattr(self,name.lower(),self._content[name]) 
			# gain correct function
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
