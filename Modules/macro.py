'''
macro.py: Ver 1.0
macro functions
'''
import numpy as np
import pandas as pd
import re
import os
import time
import ntpath
import matplotlib.pyplot as plt

import readLog
import Functions as func
import Utility as utl

import sweep
import FreqSweep
import nmr

from sweep import freqSweep as fswp
from sweep import vSweep as vswp
from FreqSweep import FreqSweep as freqswp
from nmr import nmr

#=======================================================================
def lrtzsimfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header,ftimes=1,xtimes=1,ytimes=1,rtimes=1,correctFunc=utl.gainCorrect,folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,logname=None,savename=None):
	'''
	DEPRECATED
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
	DEPRECATED
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
def lrtz_1simfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header,header_metadata=None,mainChannel='',fold=dict(),logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp',folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,savename=None):
	'''
	2020-01-20 14:49
	Fit FreqSweep type data with lrtz_1simfit method consecutively. Parse fitting result of each fit to the next fit.
	Syntax:
	-------
	result=lrtz_1simfit_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,p0,header[,header_metadata=None,mainChannel='',fold=dict(),logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp',folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,savename=None])
	Parameters:
	-----------
	device: Device code, e.g. 'h1m','TF1201'.
	filenums: File numbers to be fitted, (filelow,filehigh),fitting is done from filelow to filehigh, both filelow and filehigh can be either a list or a single item.
	fitmode,funcs1,funcs2,sharenum,folds1,folds2,frange,bounds: lrtz_1simfit fitting inputs.
	p0: Initial fitting parameters for the first file.
	header: list of str, headers corresponding to p0.
	header_metadata: list of str, metadata of fitted files read from log.
	mainChannel,fold,correctFunc,logname: File load parameters; logname is a str representing the full path of the log file.
	pMctCalib/mctBranch/Pn: parameters to update Tmct from MCT calibration and new Pn in the designated branch of melting curve. mctBranch='low' or 'high'.
	savename: str, result is written to this file.
	Returns:
	--------
	result: pandas.DataFrame, fitted results, contains filename,NMR readings, excitation info as well.
	'''
	log=pd.read_csv(logname,delim_whitespace=True)

	n=max(np.asarray(filenums[0]).size,np.asarray(filenums[1]).size) #choose the longer one's dimension as n
	lb,ub=utl.prepare_bounds(filenums,n)
	filenums=(lb,ub)

	dirname=ntpath.dirname(device)
	basename=ntpath.basename(device)
	vmkfn=np.vectorize(utl.mkFilename)#create filenames
	filerange=(vmkfn(basename,filenums[0]),vmkfn(basename,filenums[1]))
	#fetch the log associated with mems data to be fitted, use union of all ranges
	_,OrCond=utl.build_condition_dataframe(filerange,log,'Filename') #take union all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)

	index=np.linspace(0,length-1,length,dtype=int) #create index
	headerperr=[elem+'perr' for elem in header] #standard deviation headers
	Header=header_metadata+header+headerperr
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe

	ind=0
	print('Start-',end='') #progress indicator
	for i in range(0,n):
		indexl=piece[piece['Filename']==filerange[0][i]].index.values[0]
		indexu=piece[piece['Filename']==filerange[1][i]].index.values[0]
		direction=int(np.sign(indexu-indexl+0.5)) # +0.5 so that 0->1
		piecei=piece.loc[indexl:indexu:direction] # clip piece, order of rows depend on frange pairs, it can go backwards	
		for filename in piecei['Filename']:
			data=fswp(dirname+'/'+filename,mainChannel=mainChannel,fold=fold,correctFunc=correctFunc,logname=logname,normByParam=normByParam)
			if pMctCalib is not None: # update data.Tmct and its relevant
				_=data.mctC2T(pMctCalib,branch=mctBranch,Pn=Pn)
		
			#scale po according to excitation, this will scale phase as well.
			if ind==0:
				po=p0
			else:
				po=po*getattr(data,normByParam.lower())
				po[1:4]/=getattr(data,normByParam.lower()) # do not normalize d,f0,theta

			# do fit, collect: optimized parameters, std dev, residual.
			popt,_,perr,res,_,_=data.lrtz_1simfit(fitmode,funcs1,funcs2,sharenum,po,folds1=folds1,folds2=folds2,frange=frange,bounds=bounds) #fit
			po=popt/getattr(data,normByParam.lower()) #parse normalized fitted parameters to next fit, this will normalize phase as well, thus only applicable when phase and background terms are close to zero.
			po[1:4]*=getattr(data,normByParam.lower()) # do not normalize d,f0,theta

			condition=[(cn not in header_metadata) for cn in result.columns]
			result.loc[ind][condition]=np.append(popt,perr) #assign fitted values
			result.loc[ind]['Filename']=filename
			result.loc[ind]['Epoch']=data._epoch
			for name in header_metadata:
				if name not in ['Filename','Epoch']:
					result.loc[ind][name]=getattr(data,name.lower())
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
def lrtz_1simfit_fixParam_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,fix_index,fix_param,p0,header,header_metadata=None,mainChannel='',fold=dict(),logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp',folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,savename=None):
	'''
	2021-10-18 16:21
	Fit FreqSweep type data with lrtz_1simfit_fixParam method consecutively. Parse fitting result of each fit to the next fit.
	Syntax:
	-------
	result=lrtz_1simfit_fixParam_batch(device,filenums,fitmode,funcs1,funcs2,sharenum,fix_index,fix_param,p0,header[,header_metadata=None,fold=dict(),logname=None,correctFunc=utl.gainCorrect,normByParam='VLowVpp',folds1=None,folds2=None,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),pMctCalib=None,mctBranch='low',Pn=34.3934,savename=None])
	Parameters:
	-----------
	device: Device code, e.g. 'h1m','TF1201'.
	filenums: File numbers to be fitted, (filelow,filehigh),fitting is done from filelow to filehigh, both filelow and filehigh can be either a list or a single item.
	fitmode,funcs1,funcs2,sharenum,folds1,folds2,frange,bounds: lrtz_1simfit_fixParam fitting inputs.
	fix_index, fix_param, p0: Fixed and initial fitting parameters for the first file.
	header: list of str, headers corresponding to p0.
	header_metadata: list of str, metadata of fitted files read from log.
	mainChannel,fold,correctFunc,logname: File load parameters; logname is a str representing the full path of the log file.
	pMctCalib/mctBranch/Pn: parameters to update Tmct from MCT calibration and new Pn in the designated branch of melting curve. mctBranch='low' or 'high'.
	savename: str, result is written to this file.
	Returns:
	--------
	result: pandas.DataFrame, fitted results, contains filename, NMR readings, excitation info as well.
	'''
	log=pd.read_csv(logname,delim_whitespace=True)

	n=max(np.asarray(filenums[0]).size,np.asarray(filenums[1]).size) #choose the longer one's dimension as n
	lb,ub=utl.prepare_bounds(filenums,n)
	filenums=(lb,ub)

	dirname=ntpath.dirname(device)
	basename=ntpath.basename(device)
	vmkfn=np.vectorize(utl.mkFilename)#create filenames
	filerange=(vmkfn(basename,filenums[0]),vmkfn(basename,filenums[1]))

	#fetch the log associated with mems data to be fitted, use union of all ranges
	_,OrCond=utl.build_condition_dataframe(filerange,log,'Filename') #take union all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)

	index=np.linspace(0,length-1,length,dtype=int) #create index
	headerperr=[elem+'perr' for elem in header] #standard deviation headers
	Header=header_metadata+header+headerperr
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe

	ind=0
	print('Start-',end='') #progress indicator
	for i in range(0,n):
		indexl=piece[piece['Filename']==filerange[0][i]].index.values[0]
		indexu=piece[piece['Filename']==filerange[1][i]].index.values[0]
		direction=int(np.sign(indexu-indexl+0.5)) # +0.5 so that 0->1
		piecei=piece.loc[indexl:indexu:direction] # clip piece, order of rows depend on frange pairs, it can go backwards	
		for filename in piecei['Filename']:
			data=fswp(dirname+'/'+filename,mainChannel=mainChannel,fold=fold,correctFunc=correctFunc,logname=logname,normByParam=normByParam)
			if pMctCalib is not None: # update data.Tmct and its relevant
				_=data.mctC2T(pMctCalib,branch=mctBranch,Pn=Pn)
		
			#scale po according to excitation, this will scale phase as well.
			if ind==0:
				po=p0

			# do fit, collect: optimized parameters, std dev, residual.
			popt,_,perr,res,_,_=data.lrtz_1simfit_fixParam(fitmode,funcs1,funcs2,sharenum,fix_index,fix_param,po,folds1=folds1,folds2=folds2,frange=frange,bounds=bounds) #fit
			po=popt

			condition=[(cn not in header_metadata) for cn in result.columns]
			result.loc[ind][condition]=np.append(popt,perr) #assign fitted values
			result.loc[ind]['Filename']=filename
			result.loc[ind]['Epoch']=data._epoch
			for name in header_metadata:
				if name not in ['Filename','Epoch']:
					result.loc[ind][name]=getattr(data,name.lower())
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
def curve_linfit_batch(device,filenums,fitmode,header_metadata,fold={},mainChannel='',logname=None,correctFunc=utl.gainCorrect_10mVrms,corrByParam='f',reverse=False,roll_length=20,per_tol=1,pMctCalib=None,pMctBranch=None,mctBranch='low',Pn=34.3934,savename=None):
	'''
	2021-12-24 17:00
	Batch curve_linfit on files.
	Syntax:
	-------
	result = curve_linfit_batch(device,filenums,fitmode,header_metadata[,fitmode='',mainChannel='',logname=None,correctFunc=utl.gainCorrect_10mVrms,corrByParam='f',reverse=False,roll_length=20,per_tol=1,pMctCalib=None,pMctBranch=None,mctBranch='low',Pn=34.3934,savename=savename])
	Parameters:
	-----------
	device: str, device name to prepend to filenum.
	filenums: File numbers to be fitted, (filelow,filehigh),fitting is done from filelow to filehigh, both filelow and filehigh can be either a list or a single item.
	fitmode: str, only matters if it 'g' is included.
	header_metadata: list of str, metadata of fitted files read from log.
	mainChannel,fold,correctFunc,corrByParam,logname: File load parameters; logname is a str representing the full path of the log file.
	reverse,roll_length,per_tol: Functions.curve_linfit parameters.
	pMctCalib/mctBranch/Pn: parameters to update Tmct from MCT calibration and new Pn in the designated branch of melting curve. mctBranch='low' or 'high'.
	savename: str, result is written to this file.
	Returns:
	--------
	result: pandas.DataFrame, fitted results, contains filename, epoch, metadata designated by header_metadata, and fit results as well.
	'''
	log = pd.read_csv(logname, delim_whitespace=True)

	n=max(np.asarray(filenums[0]).size,np.asarray(filenums[1]).size) #choose the longer one's dimension as n
	lb,ub=utl.prepare_bounds(filenums,n)
	filenums=(lb,ub)

	dirname=ntpath.dirname(device)
	basename=ntpath.basename(device)
	vmkfn=np.vectorize(utl.mkFilename)#create filenames
	filerange=(vmkfn(basename,filenums[0]),vmkfn(basename,filenums[1]))

	#fetch the log associated with mems data to be fitted, use union of all ranges
	_,OrCond=utl.build_condition_dataframe(filerange,log,'Filename') #take union all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)

	index=np.linspace(0,length-1,length,dtype=int) #create index
	header=['slope', 'intercept', 'roll_length'] # header to store outputs
	Header=header_metadata+header
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe

	ind=0
	print('Start-',end='') #progress indicator
	for i in range(0,n):
		indexl=piece[piece['Filename']==filerange[0][i]].index.values[0]
		indexu=piece[piece['Filename']==filerange[1][i]].index.values[0]
		direction=int(np.sign(indexu-indexl+0.5)) # +0.5 so that 0->1
		piecei=piece.loc[indexl:indexu:direction] # clip piece, order of rows depend on frange pairs, it can go backwards
		for filename in piecei['Filename']:
			data=vswp(dirname+'/'+filename, fold=fold, logname=logname, mainChannel=mainChannel, correctFunc=correctFunc, corrByParam=corrByParam)
			if pMctCalib is not None: # update data.Tmct and its relevant
				_=data.mctC2T(pMctCalib,branch=mctBranch,Pn=Pn)
			
			para, fit_length = data.curve_linfit(fitmode=fitmode, reverse=reverse, roll_length=roll_length, per_tol=per_tol)
			
			condition=[(cn not in header_metadata) for cn in result.columns]
			result.loc[ind][condition]=np.append(para,fit_length) #assign fitted values
			result.loc[ind]['Filename']=filename
			result.loc[ind]['Epoch']=data._epoch
			for name in header_metadata:
				if name not in ['Filename','Epoch']:
					result.loc[ind][name]=getattr(data,name.lower())
			ind+=1
			print('-%s_%.2f%%-'%(re.sub(r'[^0-9]','',filename)[1::],(ind/length*100)),end='') #update batch progress
	print('-Finished',end='')

	if savename is not None : # save to specified file
		if os.path.isfile(savename): #file already exists
			result.to_csv(savename,sep='\t',mode='a',na_rep=np.nan,index=False,header=False,float_format='%.12e'.format)#append w/o header
		else : #file doesn't exist
			result.to_csv(savename,sep='\t',na_rep=np.nan,index=False,float_format='%.12e'.format) #create new file and save
	# return result
	return result

#=======================================================================
def nmr_1simfit_batch(device,filenums,p0,dt=2e-7,zerofillnum=0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),logpath=None,header_metadata=['Filename','_epoch','_zerofillnum','Cmct_pF'],dtLabel='dt_s',savename=None):
	'''
	Do nmr_1simfit consecutively. Each fit's optimized parameters, popt, will be transferred to the next file as an input to start fitting with.
	Syntax:
	-------
	result=nmr_1simfit_batch(device,filenums,p0[,dt=2e-7,zerofillnum=0,frange=(-np.inf,np.inf),bounds=(-np.inf,np.inf),logpath=None,dtLabel='dt_s',savename=None)
	Parameters:
	-----------
	device: device code.
	filenums: (startfilenum,endfilenum), files within range(boundary included) will be fitted. startfile can be a single item or a list of filenames to start fitting with. The same goes with endfile, e.g., (['h1m_001.dat','h1m_099.dat'],['h1m_050.dat','h1m_233.dat']) will fit 'h1m' files 001-050 and 099-233. It is allowed to have start/endfile be a single item while the other being a list, in which case the single item is shared to be the universal start/endfile.
	p0: Initial fitting parameters for fileI.
	dt: NMR FID time domain spacing, only used if log is missing dt info.
	zerofillnum: Number of zerofilling points for FID signal.
	frange: (lb,ub); Frequency range lower/upper bounds for FFT FID within which smoothing is done.
	bounds: scipy.optimize.curve_fit parameter boundaries input.
	logpath: str; NMR log file path.
	header_metadata: list of str, metadata of fitted files read from log.
	dtLabel: str; the attribute that should contain the time step info.
	savename: If exists, the output result will be saved to this file.
	Returns:
	--------
	result: pandas.DataFrame, including fitted filenames and associated meta data.
	'''
	log=pd.read_csv(logpath,delim_whitespace=True)
	# fetch the log associated with nmr data to be fitted, use union of all ranges
	dirname=ntpath.dirname(device)
	basename=ntpath.basename(device)
	vfunc=np.vectorize(utl.mkFilename)
	filerange=(vfunc(basename,filenums[0]),vfunc(basename,filenums[1]))

	_,OrCond=utl.build_condition_dataframe(filerange,log,'Filename') #take union of all ranges
	piece=log[OrCond] #these files will be fitted

	#prepare to do consecutive fit
	#create empty dataframe to store fitting results
	length=len(piece.index)
	index=np.linspace(0,length-1,length,dtype=int) #create index
	header=utl.mknmrp0Header(len(p0))
	header0=header_metadata
	headerperr=[elem+'perr' for elem in header]
	Header=header0+header+headerperr
	result=pd.DataFrame(index=index,columns=Header) #empty dataframe
	po=p0 #parameter guess for the first file
	ind=0
	print('Start-',end='') #progress indicator
	for filename in piece['Filename']:
		data=nmr(dirname+'/'+filename,zerofillnum=zerofillnum,logpath=logpath,dtLabel=dtLabel,dt=dt)
		popt,_,perr=data.fit(po,frange=frange,bounds=bounds) # use default pltflag=0
		po=popt

		condition=[(cn not in header0) for cn in result.columns]
		result.loc[ind][condition]=np.append(popt,perr) # assign fitted values
		for h in header_metadata:
			result.loc[ind][h]=getattr(data,h.lower()) # get metadata
		#result.loc[ind]['Filename']=data._filename # nmr filename
		#result.loc[ind]['Epoch']=data._epoch # nmr epoch second
		#result.loc[ind]['zerofillnum']=zerofillnum # zerofillnum
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
def tfBackground(paths,logs,mainChannels,bounds,polyDeg=9,pltflag=False,figsize=(12,5),wspace=0.3,hspace=0.3,fillstyle='full',iter_color=0,iter_marker=0,iter_linestyle=0,markeredgewidth=0.5,markersize=4,linewidth=1,legloc='upper left',bbox_to_anchor=(1,1),legsize=10):
	'''
	2019-11-21 18:21
	Calculate tuning fork background. The background is normalized to excitation=1Vpp, and the signals are rotated so that the oscillators and the reference are in-phase (phase=0 deg). The program will load every file as identified in paths, read each one's f/x/y, then scale+rotate+combine all the f/x/y to become a long f/x/y, and then fit this data using a polynomial.
	Syntax:
	-------
	px,py[,fig,axes,lines]=tfBackground(paths,logs,mainChannels,bounds[,polyDeg=9,pltflag=False,figsize=(12,5),wspace=0.3,hspace=0.3,fillstyle='full',iter_color=0,iter_marker=0,iter_linestyle=0,markeredgewidth=0.5,markersize=4,linewidth=1,legloc='upper left',bbox_to_anchor=(1,1),legsize=10])

	Parameters:
	-----------
	paths: list, a list of all files that are going to be used to fit the background, [path0,path1,...]
	logs: list, a list of the log files corresponding to every file in the paths list, [log0,log1,...]
	mainChannels: list, a list of mainChannel strings corresponding to every file in the paths list, [mainChannel0,mainChannel1,...]
	bounds: list, a list of bounds corresponding to every file in the paths list, [bounds0,bounds1,...]. Each bounds#=(lb,ub) is a range of f/x/y that will be included from that file to be part of the macro fitting. e.g. bounds1=([1100,13950,34550],[8950,29550,np.inf]) means that for the file identified by path1, the frequency range [1100,8950]U[13950,29550]U[34550,inf] will be selected, as well as its associated x and y, which will later be combined with the data from the other path# files to become the data for fitting.
	polyDeg: int, the degree of the polynomial fit.
	pltflag: boolean, whether to plot the fitting results.
	figsize,wspace,hspace,fillstyle,iter_color,iter_marker,iter_linestyle,markeredgewidth,markersize,linewidth,legloc,bbox_to_anchor,legsize: plot parameters.
	Returns:
	--------
	px: numpy.array, fitted polynomial parameters for the x-channels.
	py: numpy.array, fitted polynomial parameters for the y-channels.
	fig,axes,lines: plot handles.
	'''
	f=np.array([])
	x=np.array([])
	y=np.array([])
	for path,log,mc,b in zip(paths,logs,mainChannels,bounds):
		data=fswp(path,logname=log,mainChannel=mc,normByParam='voltagevpp'+mc)
		_,od=utl.build_condition_series(b,data.f)
		phase=getattr(data,'phase'+mc)/180*np.pi
    
		f0=data.f[od]
		x0=data.nx[od]*np.cos(phase)-data.ny[od]*np.sin(phase) # all measurements are normalized to 1Vpp excitation, rotated to 0deg.
		y0=data.nx[od]*np.sin(phase)+data.ny[od]*np.cos(phase) 
    
		f=np.concatenate( (f,f0),axis=0 )
		x=np.concatenate( (x,x0),axis=0 )
		y=np.concatenate( (y,y0),axis=0 )

	px=np.polyfit(f,x,polyDeg)
	py=np.polyfit(f,y,polyDeg)

	iter_color=iter_color
	iter_marker=iter_marker
	iter_linestyle=iter_linestyle
	lines=[]
	if pltflag:
		fig,axes=plt.subplots(1,2,figsize=figsize)
		[ax1,ax2]=axes
		fig.subplots_adjust(wspace=wspace,hspace=hspace)
		for path,log,mc,b in zip(paths,logs,mainChannels,bounds):
			data=fswp(path,logname=log,mainChannel=mc,normByParam='voltagevpp'+mc)
			phase=getattr(data,'phase'+mc)/180*np.pi
			V=getattr(data,'voltagevpp'+mc)
			bx0=np.polyval(px,data.f)
			by0=np.polyval(py,data.f)
			bx=(bx0*np.cos(phase)+by0*np.sin(phase))*V
			by=(-bx0*np.sin(phase)+by0*np.cos(phase))*V
        
			linex1=ax1.plot(data.f,data.x,color=utl.colorCode(iter_color%utl.lencc()),marker=utl.markerCode(iter_marker%23),fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linestyle='',label=data._filename)
			linex2=ax2.plot(data.f,data.y,color=utl.colorCode(iter_color%utl.lencc()),marker=utl.markerCode(iter_marker%23),fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=markersize,linestyle='',label=data._filename)
			liney1=ax1.plot(data.f,bx,color=utl.colorCode(iter_color%utl.lencc()),fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=0,linestyle=utl.linestyleCode(iter_linestyle%4),linewidth=linewidth,label=data._filename)
			liney2=ax2.plot(data.f,by,color=utl.colorCode(iter_color%utl.lencc()),fillstyle=fillstyle,markeredgewidth=markeredgewidth,markersize=0,linestyle=utl.linestyleCode(iter_linestyle%4),linewidth=linewidth,label=data._filename)
        
			ax1.set_xlabel('f (Hz)')
			ax1.set_ylabel('x (Vrms)')
			ax2.set_xlabel('f (Hz)')
			ax2.set_ylabel('y (Vrms)')
			ax2.legend(loc=legloc,bbox_to_anchor=bbox_to_anchor,prop={'size':legsize})
        
			iter_color+=1
			iter_marker+=1
			iter_linestyle+=1
        
			linex1.append(*linex2)
			linex1.append(*liney1)
			linex1.append(*liney2)
			lines.append(linex1)
        
		plt.show()
	if pltflag:
		return px,py,fig,axes,lines
	else:
		return px,py
#=======================================================================

