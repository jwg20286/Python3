import numpy as np
import pandas as pd
import datetime

#=======================================================================
class mctLog(object):
	'''
	MCT log class.
	The read file contains a machine-reading-unfriendly header row. It isskipped here by setting skiprows.
	Syntax:
	-------
	dataframe=mctLog(filename[,header=['Date','Time','No','Tmct_mK','Pmct_bar','Pn_bar','Cmct_pF','Loss_nS','B_kG','GHS-G1_bar','AH-V_V','AH-AV_s','AH-AL_s'],skiprows=1])
	Parameters:
	-----------
	filename: mct log file name.
	header: the input header used to replace the header row in the source file.
	skiprows: number of rows to skip above the source file.
	Returns:
	--------
	self.filename: filename.
	self.log: pandas.DataFrame, column names are set by input.
	self.date: str, date of '%m/%d/%Y' format.
	self.time: str, time of '%H:%M:%S' format.
	self.No: iteration number.
	self.Tmct: mct temperature in mK.
	self.Pmct: mct pressure in bar.
	self.Pn: the Neel pressure, in bar, used to do the Pmct->Tmct conversion.
	self.Cmct: mct capacitance in pF.
	self.loss: mct loss in nS.
	self.B: magnet current in kG, AMI M420 reading.
	self.ghsG1: GHS-G1 pressure in bar.
	self.datetime: datetime.datetime.
	self.epoch: epoch seconds calculated from self.datetime.
	'''
	def __init__(self,filename,header=['Date','Time','No','Tmct_mK','Pmct_bar','Pn_bar','Cmct_pF','Loss_nS','B_kG','GHS-G1_bar','AH-V_V','AH-AV_s','AH-AL_s'],skiprows=1):
		self.filename=filename
		self.log=pd.read_csv(filename,delim_whitespace=True,header=None,skiprows=skiprows,names=header)
		self.date=self.log['Date'].values
		self.time=self.log['Time'].values
		self.No=self.log['No'].values
		self.Tmct=self.log['Tmct_mK'].values
		self.Pmct=self.log['Pmct_bar'].values
		self.Pn=self.log['Pn_bar'].values
		self.Cmct=self.log['Cmct_pF'].values
		self.loss=self.log['Loss_nS'].values
		self.B=self.log['B_kG'].values
		self.ghsG1=self.log['GHS-G1_bar'].values
#-----------------------------------------------------------------------
		vstrptime=np.vectorize(datetime.datetime.strptime)
		self.datetime=vstrptime(self.date+' '+self.time,'%m/%d/%Y %H:%M:%S') #array of datetime.datetime dtype

		findEpoch=lambda dt: (dt-datetime.datetime(1970,1,1,0,0,0)).total_seconds() #find epoch seconds of dt, which is of datetime.datetime type.
		vfindEpoch=np.vectorize(findEpoch)
		self.epoch=vfindEpoch(self.datetime)
#=======================================================================
class nmrLog(object):
	'''
	NMR Log class.
	Syntax:
	-------
	nl=nmrLog(filename)
	Parameters:
	-----------
	filename: nmr log filename.
	Returns:
	--------
	self.filename: nmr log filename.
	self.log: pandas.DataFrame, log content.
	self.date: str, date of '%m/%d/%Y' format.
	self.time: str, time of '%H:%M:%S' format.
	self.datetime: datetime.datetime.
	self.epoch: epoch seconds calculated from self.datetime.
	self.Cmct: mct capacitance in pF.
	self.Tmct: mct temperature in mK.
	self.plm: PLM-3 display.
	self.nmrabs: numpy.array, sum of nmr file absolute value.
	self.nmrfilter: numpy.array, labview output of filtered self.nmrabs, the filter is applied to the fft with a pass window set by the nmr labview program.
	self.nmrfilename: numpy.array, all recorded NMR filenames.
	'''
	def __init__(self,filename):
		self.filename=filename
		self.log=pd.read_csv(filename,delim_whitespace=True,index_col=False)
		self.date=self.log['Date'].values
		self.time=self.log['Time'].values
		self.Cmct=self.log['CMCTpF'].values
		self.Tmct=self.log['TMCTmK'].values
		self.plm=self.log['PLMDisplay'].values
		self.nmrabs=self.log['NMRAbs'].values
		self.nmrfilter=self.log['NMRFilter'].values
		self.nmrfilename=self.log['NMRFilename'].values
#-----------------------------------------------------------------------
		vstrptime=np.vectorize(datetime.datetime.strptime)
		self.datetime=vstrptime(self.date+' '+self.time, '%m/%d/%Y %H:%M:%S') #array of datetime.datetime dtype

		findEpoch=lambda dt: (dt-datetime.datetime(1970,1,1,0,0,0)).total_seconds() #find epoch seconds of dt, which is of datetime.datetime type
		vfindEpoch=np.vectorize(findEpoch)
		self.epoch=vfindEpoch(self.datetime)
#=======================================================================
class freqSweepLog(object):
	'''
	FreqSweep Log class.
	Syntax:
	-------
	fswpl=freqSweepLog(filename)
	Parameters:
	-----------
	filename: FreqSweep log filename.
	Returns:
	--------
	self.filename: FreqSweep log filename.
	self.log: pandas.DataFrame, log content.
	self.date: str, date of '%m/%d/%Y' format.
	self.time: str, time of '%H:%M:%S' format.
	self.datetime: datetime.datetime.
	self.epoch: epoch seconds calculated from self.datetime.
	self.fswpfilename: FreqSweep filenames.
	self.batchnum: numpy.array, number of sweep in batch.
	self.frange: numpy.array, sweep frequency range.
	self.flow: numpy.array, frequency low bound in Hz.
	self.fhigh: numpy.array, frequency high bound in Hz.
	self.swptime: numpy.array, sweep time per point in ms.
	self.numpts: numpy.array, number of points per sweep.
	self.LIsens: numpy.array, lock-in sensitivity in V.
	self.LITconst: numpy.array, lock-in time constant.
	self.LIphase: numpy.array, lock-in phase.
	self.updown: numpy.array, direction of frequency sweep.
	self.vl: numpy.array, low frequency excitation in V.
	self.nmrfilename: numpy.array, all recorded NMR filenames.
	self.plm: numpy.array, PLM-3 display.
	self.nmrfilter: numpy.array, labview output of filtered self.nmrabs, the filter is applied to the fft with a pass window set by the nmr labview program.
	'''
	def __init__(self,filename):
		self.filename=filename
		self.log=pd.read_csv(filename,delim_whitespace=True,index_col=False)
		self.date=self.log['Date'].values
		self.time=self.log['Time'].values
		self.fswpfilename=self.log['Filename'].values
		self.batchnum=self.log['BatchNum'].values
		self.frange=self.log['FreqRange'].values
		self.flow=self.log['FreqLow'].values
		self.fhigh=self.log['FreqHigh'].values
		self.swptime=self.log['SwpTime'].values
		self.numpts=self.log['NumPoints'].values
		self.LIsens=self.log['LISensitivity'].values
		self.LITconst=self.log['LITimeConstant'].values
		self.LIphase=self.log['LIPhase'].values
		self.updown=self.log['UpDown'].values
		self.vl=self.log['VLowVpp'].values
		self.nmrfilename=self.log['NMRFilename'].values
		self.plm=self.log['PLMDisplay'].values
		self.nmrfilter=self.log['FilteredAbsSum'].values
#-----------------------------------------------------------------------
		vstrptime=np.vectorize(datetime.datetime.strptime)
		self.datetime=vstrptime(self.date+' '+self.time, '%m/%d/%Y %H:%M:%S') #array of datetime.datetime dtype
		findEpoch=lambda dt: (dt-datetime.datetime(1970,1,1,0,0,0)).total_seconds() #find epoch seconds of dt, which is of datetime.datetime type
		vfindEpoch=np.vectorize(findEpoch)
		self.epoch=vfindEpoch(self.datetime)
#=======================================================================
class sweepLog(object):
	'''
	sweep log class.
	The log column headers are converted to lower cases when assigned to attributes, without underscore in front. Other attributes start with one underscore.
	Syntax:
	-------
	swpl=swpLog(filename)
	Returns:
	--------
	self._filename: str, sweep log filename.
	self._content: pandas.DataFrame, entire log content.
	self.item: pandas.Series, each item is one column with the same name (but lower case) in the content.
	self._datetime: datetime.datetime, combination of date and time.
	self._epoch: float, epoch seconds calculated from datetime.
	'''
	def __init__(self,filename):
		self._filename=filename
		self._content=pd.read_csv(filename,delim_whitespace=True,index_col=False)
		col_names=self._content.columns.tolist()
		for name in col_names:
			# assign pandas.Series to attributes based on name
			setattr(self,name.lower(),self._content[name])
#-----------------------------------------------------------------------
		# assign self._datetime
		# only runs if 'date' and 'time' exists (case insensitive)
		lower_col_names=[x.lower() for x in self._content.columns]
		if 'date' and 'time' in lower_col_names:
			vstrptime=np.vectorize(datetime.datetime.strptime)
			self._datetime=vstrptime(self.date+' '+self.time, '%m/%d/%Y %H:%M:%S') # array of datetime.datetime dtype

		# assign self._epoch.
			findEpoch=lambda dt: (dt-datetime.datetime(1970,1,1,0,0,0)).total_seconds() #find epoch seconds of dt, which is of datetime.datetime type
			vfindEpoch=np.vectorize(findEpoch)
			self._epoch=vfindEpoch(self._datetime)
#=======================================================================
