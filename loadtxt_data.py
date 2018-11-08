import numpy as np
import pandas as pd


def readtxtNumpyRow(filename,rows):
	data = np.loadtxt(filename,dtype=np.float32,delimiter=',')
	return(data[[row-1 for row in rows],:])

def readtxtNumpyColumn(filename,columns):
	data=np.loadtxt(filename,dtype=np.float32,delimiter=',')
	return(data[:,[column-1 for column in columns]])

def readtxtpandasrow(filename,rows):
	data = pd.read_table(filename,header=None,sep=',')
	return(data.ix[[row-1 for row in rows]])

def readtxtpandascolumn(filename,columns):
	data = pd.read_table(filename,header=None,sep=',')
	return(data.ix[:,[column-1 for column in columns]])

def readtxtopenrow(filename,rows):
	data=[]
	with open(filename,'r') as f:
		lines = f.readlines()
		for row in rows:
			data.append(lines[row-1].strip().split(','))
	return(data)

def readtxtopencolumn(filename,columns):
	data = []
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			tmp=[]
			for column in columns:
				tmp.append(line.split(',')[column-1])
			data.append(tmp)
	return(data)

if __name__ =="__main__":
	# print(readtxtNumpyRow('postVPCJointPath16.txt',[1,3]))
	# print(readtxtNumpyColumn('postVPCJointPath16.txt',[2,4]))
	# print(readtxtpandasrow('postVPCJointPath16.txt',[1,3]))
	# print(readtxtpandascolumn('postVPCJointPath16.txt',[1,3]))
	# print(readtxtopenrow('postVPCJointPath16.txt',[1,3]))
	print(readtxtopencolumn('postVPCJointPath16.txt',[2,4]))