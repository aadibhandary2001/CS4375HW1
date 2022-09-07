import pandas as pd
import numpy as np

def preceptron_helper(n):
	if n>=0:
		return 0
	else:
		return 1
		
def preceptron(x_frame,y_frame,w_list,b_const):
	M=len(x_frame)
	x_np=x_frame.to_numpy()
	y_np=y_frame.to_numpy()
	w_np=np.array(w)
	val_arry=[]
	return 3

def loss_func(x_frame,y_frame,w_list,b_const):
	M=len(x_frame)
	x_np=x_frame.to_numpy()
	y_np=y_frame.to_numpy()
	w_np=np.array(w)
	val_arry=[]
	for m in range(0,M):
		dot_prod=np.dot(x_np[m],w_np)
		val=-1*y_np[m]*(dot_prod+b_const)
		val=max(0,val)
		val_arry.append(val)
	return (1/M)*sum(val_arry)
	
if __name__ == '__main__':
	w=[1,2,3,4]
	b=3
	colnames=["x1","x2","x3","x4","y"]
	data=pd.read_csv("perceptron.csv",names=colnames,header=None)
	x=data[["x1","x2","x3","x4"]]
	y=data[["y"]]
	loss=loss_func(x,y,w,b)
