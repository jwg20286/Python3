'''
macro.py: Ver 1.0
macro functions
'''
import numpy as np
import pandas as pd
import re
import os
import time

import readLog
import Functions as func
import Utility as utl

import sweep
import FreqSweep
import nmr

from sweep import freqSweep as fswp
from FreqSweep import FreqSweep as freqswp
from nmr import nmr

#=======================================================================
def lrtzsimfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header,ftimes=1,xtimes=1,ytimes=1,rtimes=1,correctFunc=utl.gainCorrect,folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,logname=None,savename=None):
	'''
	Fit FreqSweep type data with lrtz1simfit method consecutively. Parse fitting result of each fit to the next fit.
	Syntax:
	-------
	result=lrtzsimfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header,logname=None[,ftimes=1,xtimes=1,ytimes=1,rtimes=1,correctFunc=utl.gainCorrect,folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,savename=None])
	Parameters:
	-----------
	device: Device code, e.g. 'h1m','TF1201'.
	filenums: File numbers to be fitted, (filelow,filehigh),fitting is done from filelow to filehigh, both filelow and filehigh can be either a list or a single item.
	fitmode,funcs1,funcs2,sharenum,folds1,folds2,frange,bounds: lrtz1simfit fitting inputs.
	p0: Initial fitting parameters for the first file.
	header: Headers corresponding to p0.
	f/x/y/rtimes,correctFunc,logname: File load parameters.
	pMctCalib/mctBranch/Pn: parameters to update Tmct from MCT calibration and new Pn in the designated branch of melting curve. mctBranch='low' or 'high'.
	savename: Result is written to this file.
	Returns:
	--------
	result: pandas.DataFrame, fitted results, contains filename,NMR readings, excitation info as well.
	'''
	log=pd.read_csv(logname,delim_whitespace=True)

	vfunc=np.vectorize(utl.mkFilename)#create filenames
	filerange=(vfunc(device,filenums[0]),vfunc(device,filenums[1]))
	#fetch the log associated with mems data to be fitted, use union of all ranges
	_,OrCond=utl.build_condition_dataframe(filerange,log,'Filename') #take union all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)
	index=np.linspace(0,length-1,length,dtype=int) #create index
	header0=['Filename','Epoch','BatchNum','VLow','Tmct','Tmm','NMRFilename','PLMDisplay','FilteredAbsSum']
	headerperr=[elem+'perr' for elem in header] #standard deviation headers
	Header=header0+header+headerperr
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe

	ind=0
	print('Start-',end='') #progress indicator
	for filename in piece['Filename']:
		data=freqswp(filename,ftimes=ftimes,xtimes=xtimes,ytimes=ytimes,rtimes=rtimes,correctFunc=correctFunc,logname=logname)
		if pMctCalib is not None: # update data.Tmct and its relevant
			_=data.mctC2T(pMctCalib,branch=mctBranch,Pn=Pn)
		
		#scale po according to excitation, this will scale phase as well.
		if ind==0:
			po=p0
		else:
			po=po*data.vl 

		# do fit, collect: optimized parameters, std dev, residual.
		popt,_,perr,res,_,_=data.lrtz1simfit(fitmode,funcs1,funcs2,sharenum,p0,folds1=folds1,folds2=folds2,frange=frange,bounds=bounds) #fit
		po=popt/data.vl #parse normalized fitted parameters to next fit, this will normalize phase as well, thus only applicable when phase is close to zero.

		condition=[(cn not in header0) for cn in result.columns]
		result.loc[ind][condition]=np.append(popt,perr) #assign fitted values
		result.loc[ind]['Filename']=filename
		result.loc[ind]['Epoch']=data.epoch
		result.loc[ind]['BatchNum']=data.batchnum
		result.loc[ind]['VLow']=data.vl
		result.loc[ind]['Tmct']=data.avgTmct
		result.loc[ind]['Tmm']=data.Tmm
		result.loc[ind]['NMRFilename']=data.nmrfilename
		result.loc[ind]['PLMDisplay']=data.plm
		result.loc[ind]['FilteredAbsSum']=data.nmrfilter
		ind+=1
		print('-%s_%.2f%%-'%(re.sub(r'[^0-9]','',filename)[1::],(ind/length*100)),end='') #update batch progress
	print('-Finished',end='')
	
	if savename is not None: #save to specified file
		if os.path.isfile(savename): #file already exists
			result.to_csv(savename,sep='\t',mode='a',na_rep=np.nan,index=False,header=False,float_format='%.12e'.format)#append w/o header
		else: #file doesn't exist
			result.to_csv(savename,sep='\t',na_rep=np.nan,index=False,float_format='%.12e'.format) #create new file and save
	return result
#=======================================================================
def nmrsimfit_batch(filenums,p0,window_size,order,device='NMR',tstep=2e-7,zerofillnum=0,sfrange=(-np.inf,np.inf),deriv=0,rate=1,bounds=(-np.inf,np.inf),logname='NLog_001.dat',savename=None):
	'''
	Do nmr1simfit consecutively. Each fit's optimized parameters, popt, will be transferred to the next file as an input to start fitting with.
	Syntax:
	-------
	result=nmrsimfit_batch(filenums,p0,window_size,order[,device='NMR',tstep=2e-7,zerofillnum=0,sfrange=(-inf,inf),deriv=0,rate=1,bounds=(-inf,inf),logname='NLog_001.dat',savename=None)
	Parameters:
	-----------
	filenums: (startfilenum,endfilenum), files within range(boundary included) will be fitted. startfile can be a single item or a list of filenames to start fitting with. The same goes with endfile, e.g., (['h1m_001.dat','h1m_099.dat'],['h1m_050.dat','h1m_233.dat']) will fit 'h1m' files 001-050 and 099-233. It is allowed to have start/endfile be a single item while the other being a list, in which case the single item is shared to be the universal start/endfile.
	p0: Initial fitting parameters for fileI.
	window_size,order,deriv,rate: savitzky_golay smooth method input parameters.
	device: device code.
	tstep: NMR FID time domain spacing.
	zerofillnum: Number of zerofilling points for FID signal.
	sfrange: Frequency range for FFT FID within which smoothing is done.
	bounds: scipy.optimize.curve_fit parameter boundaries input.
	logname: NMR log file name.
	savename: If exists, the output result will be saved to this file.
	Returns:
	--------
	result: pandas.DataFrame, including fitted filenames and associated meta data.
	'''
	log=pd.read_csv(logname,delim_whitespace=True)
	# fetch the log associated with nmr data to be fitted, use union of all ranges
	vfunc=np.vectorize(utl.mkFilename)
	filerange=(vfunc(device,filenums[0]),vfunc(device,filenums[1]))

	_,OrCond=utl.build_condition_dataframe(filerange,log,'NMRFilename') #take union of all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)
	index=np.linspace(0,length-1,length,dtype=int) #create index
	header=utl.mknmrp0Header(len(p0))
	header0=['NMRFilename','NMREpoch','zerofillnum']
	headerperr=[elem+'perr' for elem in header]
	Header=header0+header+headerperr
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe
	po=p0 #parameter guess for the first file
	ind=0
	print('Start-',end='') #progress indicator
	for filename in piece['NMRFilename']:
		data=nmr(filename,tstep=tstep,zerofillnum=zerofillnum)
		popt,_,perr=data.fit(po,window_size,order,sfrange=sfrange,deriv=deriv,rate=rate,bounds=bounds) # use default pltflag=0
		po=popt

		condition=[(cn not in header0) for cn in result.columns]
		result.loc[ind][condition]=np.append(popt,perr) # assign fitted values
		result.loc[ind]['NMRFilename']=filename # nmr filename
		result.loc[ind]['NMREpoch']=data.epoch # nmr epoch second
		result.loc[ind]['zerofillnum']=zerofillnum # zerofillnum
		ind+=1
		print('-%s-%.2f%%-'%(re.sub(r'[^0-9]','',filename)[0::],ind/length*100),end='') #update progress
	print('-Finished',end='')

	if savename is not None : # save to specified file
		if os.path.isfile(savename): #file already exists
			result.to_csv(savename,sep='\t',mode='a',na_rep=np.nan,index=False,header=False,float_format='%.12e'.format)#append w/o header
		else : #file doesn't exist
			result.to_csv(savename,sep='\t',na_rep=np.nan,index=False,float_format='%.12e'.format) #create new file and save
	return result
#=======================================================================
def logMean_single(logname,frange,colname,droplabels=None,dropAxis=1,drop_track=True,meanAxis=0):
	'''
	2017-06-22 15:13
	Average specific part of a log file and output the average and standard deviation as two separate pandas.series.
	Syntax:
	-------
	pMean,pStd=logMean_single(logname,frange,colname,droplabels=None,dropAxis=1,drop_track=True,meanAxis=0)
	Parameters:
	-----------
	logname: log data file name.
	frange: (lowb,highb) format range, lowb is a single item, or lowb is a list of all the lowbounds, corresponding to some high bounds stored in high b. lowb==highb is allowed. The data within this range will be averaged.
	colname: column name under which the frange is applied, and the conditions are built.
	droplabels: single label or list-like, labels to be excluded from averaging.
	dropAxis: dataframe.drop axis, 1 means drop a column.
	drop_track: boolean, if True, include the 1st&last rows of the dropped columns as extra columns in output.
	meanAxis: dataframe.mean/std axis, 0 means average the column values.
	Returns:
	--------
	pMean: pandas.Series, mean of the chosen piece of data, numeric only, if drop_track==True, the rest of the columns in logfile will also be included.
	pStd: pandas.Series, standard deviation of the chosen piece of data, numeric only.
	'''
	log=pd.read_csv(logname,delim_whitespace=True)
	pMean,pStd=utl.partial_dataframe_mean(frange,log,colname,droplabels=droplabels,dropAxis=dropAxis,meanAxis=meanAxis)
#-----------------------------------------------------------------------
	if drop_track is True : #keep track of droppled labels
		_,_,clus_1st_last=utl.partial_dataframe_cluster(frange,log,colname,dropIndex=pMean.index)
		meanSeries=pd.concat([clus_1st_last,pMean])#extend pMean
		stdSeries=pd.concat([clus_1st_last,pStd])#extend pStd

		return meanSeries,stdSeries
#-----------------------------------------------------------------------
	return pMean,pStd
#=======================================================================
def logMean(logname,frange,colname,droplabels=None,dropAxis=1,drop_track=True,meanAxis=0,saveflag=False,savename=None):
	'''
	2017-06-22 15:18
	Average specific part of a log file and output the average and standard deviation as two dataframes.
	Syntax:
	-------
	df_Mean,df_Std=logMean(logname,frange,colname,droplabels=None,dropAxis=1,meanAxis=0,saveflag=False,savename=None)
	Parameters:
	-----------
	logname: log data file name.
	frange: (lowb,highb) format range, lowb is a single item, or lowb is a list of all the lowbounds, corresponding to some high bounds stored in high b. lowb==highb is allowed. The data within this range will be averaged.
	colname: column name under which the frange is applied, and the conditions are built.
	droplabels: single label or list-like, labels to be excluded from averaging.
	dropAxis: dataframe.drop axis, 1 means drop a column.
	drop_track: boolean, if True, include the 1st&last rows of the dropped columns as extra columns in output.
	meanAxis: dataframe.mean/std axis, 0 means average the column values.
	saveflag: boolean, save outputs if true. Mean and standard deviation will be saved to 2 different files, which with a filename similar to logname, but an extra tail appended, the tail for mean is '_MeanCurrentDate', for standard deviation is '_StdCurrentDate'.
	savename: list of str, [savename_mean, savename_std], df_Mean will be saved to savename_mean, and df_Std will be saved to savename_Std. When savename is given, saveflag is forced to be True.
	Returns:
	--------
	df_Mean: pandas.DataFrame, mean of the chosen piece of data, numeric only, if drop_track==True, the rest of the columns in logfile will also be included.
	df_Std: pandas.DataFrame, standard deviation of the chosen piece of data, numeric only.
	'''
	n=max(np.asarray(frange[0]).size,np.asarray(frange[1]).size)
	lb,ub=utl.prepare_bounds(frange,n)
	list_Mean=[]
	list_Std=[]
	for llbb,uubb in zip(lb,ub):
		sMean,sStd=logMean_single(logname,(llbb,uubb),colname,droplabels=droplabels,dropAxis=dropAxis,drop_track=drop_track,meanAxis=meanAxis)	
		list_Mean+=[sMean]
		list_Std+=[sStd]

	df_Mean=pd.concat(list_Mean,axis=1).transpose() #return dataframe
	df_Std=pd.concat(list_Std,axis=1).transpose()
#-----------------------------------------------------------------------
	#create savename_mean and savename_std
	if savename is None and saveflag is True : #savename not sepcified => autogenerate savename
		currentDate=time.strftime('%y%m%d')#make sure Mean and Std use the same date
		logname_split=os.path.splitext(logname) #split name and extension
		# save df_Mean to one file and df_Std to another
		savename_mean=logname_split[0]+'_Mean'+currentDate+logname_split[1] # append '_MeanYrMonthDay' to logname as savename for mean output
		savename_std=logname_split[0]+'_Std'+currentDate+logname_split[1] # append '_StdYrMonthDay' to logname as savename for mean output
	elif savename is not None : # if savename specified, do not care saveflag anymore
		savename_mean=savename[0]
		savename_std=savename[1]
		saveflag=True # froce flag to be True if not yet True
#-----------------------------------------------------------------------
	# save df_Mean and df_Std to new files, or append to existing files
	if saveflag is True:	
		if os.path.isfile(savename_mean): #file already exists
			#df_Mean.to_csv(savename_mean,sep='\t',mode='a',na_rep=np.nan,index=False,header=False,float_format='%.12e'.format)#append w/o header # looks like float_format is causing problem where a bunch of % or $ are saved instead of the actual data
			df_Mean.to_csv(savename_mean,sep='\t',mode='a',na_rep=np.nan,index=False,header=False)#append w/o header
		else : #file doesn't exist
			df_Mean.to_csv(savename_mean,sep='\t',na_rep=np.nan,index=False) #create new file and save	
			# save df_Std to another file

		if os.path.isfile(savename_std): #file already exists
			df_Std.to_csv(savename_std,sep='\t',mode='a',na_rep=np.nan,index=False,header=False)
		else : #file doesn't exist
			df_Std.to_csv(savename_std,sep='\t',na_rep=np.nan,index=False) #create new file and save
#-----------------------------------------------------------------------
	return df_Mean,df_Std
#=======================================================================
def lrtz_1simfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header,header_metadata=None,fold=dict(),directory='./',logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp',folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,savename=None):
	'''
	2019-01-09 20:15
	Fit FreqSweep type data with lrtz_1simfit method consecutively. Parse fitting result of each fit to the next fit.
	Syntax:
	-------
	result=lrtz_1simfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header[,header_metadata=None,ftimes=1,xtimes=1,ytimes=1,rtimes=1,correctFunc=utl.gainCorrect,folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,logname=None,savename=None])
	Parameters:
	-----------
	device: Device code, e.g. 'h1m','TF1201'.
	filenums: File numbers to be fitted, (filelow,filehigh),fitting is done from filelow to filehigh, both filelow and filehigh can be either a list or a single item.
	fitmode,funcs1,funcs2,sharenum,folds1,folds2,frange,bounds: lrtz_1simfit fitting inputs.
	p0: Initial fitting parameters for the first file.
	header: list of str, headers corresponding to p0.
	header_metadata: list of str, metadata of fitted files read from log.
	f/x/y/rtimes,correctFunc,logname: File load parameters; logname is a str representing the full path of the log file.
	directory: str, directory of the folder that contains the target data files.
	pMctCalib/mctBranch/Pn: parameters to update Tmct from MCT calibration and new Pn in the designated branch of melting curve. mctBranch='low' or 'high'.
	savename: str, Result is written to this file.
	Returns:
	--------
	result: pandas.DataFrame, fitted results, contains filename,NMR readings, excitation info as well.
	'''
	log=pd.read_csv(logname,delim_whitespace=True)

	vfunc=np.vectorize(utl.mkFilename)#create filenames
	filerange=(vfunc(device,filenums[0]),vfunc(device,filenums[1]))
	#fetch the log associated with mems data to be fitted, use union of all ranges
	_,OrCond=utl.build_condition_dataframe(filerange,log,'Filename') #take union all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)
	index=np.linspace(0,length-1,length,dtype=int) #create index
	#header0=['Filename','Epoch','BatchNum','VLow','Tmct','Tmm','NMRFilename','PLMDisplay','FilteredAbsSum']
	header0=header_metadata
	headerperr=[elem+'perr' for elem in header] #standard deviation headers
	Header=header0+header+headerperr
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe

	ind=0
	print('Start-',end='') #progress indicator
	for filename in piece['Filename']:
		data=fswp(directory+filename,fold=fold,correctFunc=correctFunc,logname=logname,normByParam=normByParam)
		if pMctCalib is not None: # update data.Tmct and its relevant
			_=data.mctC2T(pMctCalib,branch=mctBranch,Pn=Pn)
		
		#scale po according to excitation, this will scale phase as well.
		if ind==0:
			po=p0
		else:
			po=po*getattr(data,normByParam.lower())

		# do fit, collect: optimized parameters, std dev, residual.
		popt,_,perr,res,_,_=data.lrtz_1simfit(fitmode,funcs1,funcs2,sharenum,p0,folds1=folds1,folds2=folds2,frange=frange,bounds=bounds) #fit
		po=popt/getattr(data,normByParam.lower()) #parse normalized fitted parameters to next fit, this will normalize phase as well, thus only applicable when phase and background terms are close to zero.

		condition=[(cn not in header0) for cn in result.columns]
		result.loc[ind][condition]=np.append(popt,perr) #assign fitted values
		result.loc[ind]['Filename']=filename
		result.loc[ind]['Epoch']=data._epoch
		for name in header0:
			if name not in ['Filename','Epoch']:
				result.loc[ind][name]=getattr(data,name.lower())
		'''
		#quoted 
		result.loc[ind]['Epoch']=data.epoch
		result.loc[ind]['BatchNum']=data.batchnum
		result.loc[ind]['VLow']=data.vl
		result.loc[ind]['Tmct']=data.avgTmct
		result.loc[ind]['Tmm']=data.Tmm
		result.loc[ind]['NMRFilename']=data.nmrfilename
		result.loc[ind]['PLMDisplay']=data.plm
		result.loc[ind]['FilteredAbsSum']=data.nmrfilter
		'''
		ind+=1
		print('-%s_%.2f%%-'%(re.sub(r'[^0-9]','',filename)[1::],(ind/length*100)),end='') #update batch progress
	print('-Finished',end='')
	
	if savename is not None: #save to specified file
		if os.path.isfile(savename): #file already exists
			result.to_csv(savename,sep='\t',mode='a',na_rep=np.nan,index=False,header=False,float_format='%.12e'.format)#append w/o header
		else: #file doesn't exist
			result.to_csv(savename,sep='\t',na_rep=np.nan,index=False,float_format='%.12e'.format) #create new file and save
	return result
#=======================================================================

