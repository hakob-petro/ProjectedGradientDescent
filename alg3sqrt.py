import numpy as np
import math
from datetime import datetime
import random
from n20m200 import n,m,C,b

def norm(x):
	for i in range(n):
		if(x[i]<=0):x[i]=1e-8
	if(np.linalg.norm(x)>R):
		exit()

def G(g,b,x,tr):
	if(tr=='grad'):
		if(tg=='linear'):t=g
		if(tg=='abs'):t=g*np.sign(x)
		if(tg=='sqr'):t=g*np.multiply(2,x)
	if(tr=='g'):
		if(tg=='linear'):t=sum(g*x)-b
		if(tg=='abs'):t=sum(g*abs(x))-b
		if(tg=='sqr'):t=sum(g*x**2)-b
	return t

# Modification.
def Grad_g_m(x):
	global M,v
	for i in range(len(C)):
		M=np.linalg.norm(C[i])
		v=G(C[i],b[i],x,'g')
		if(v/M>eps):
			return G(C[i],None,x,'grad')
	return [None]

# Standard.
def Grad_g_s(x):
	global M
	v=G(C[0],b[0],x,'g')
	gm=C[0]
	for i in range(len(C)):
		t=G(C[i],b[i],x,'g')
		if(t>v):
			v=t
			gm=C[i]
	M=np.linalg.norm(gm)
	if(v/M>eps):
		return G(gm,None,x,'grad')
	return [None]

def f(x):
	# return sum(i**2 for i in x)
	# return sum((i+1)*x[i]**2+math.exp(-(i+1)*x[i]) for i in range(n))
	return -sum(math.sqrt(i) for i in x)/n
	# return -sum(math.log(i) for i in x)/n

def Grad_f(x):
	# return [2*i for i in x]
	# return [2*(i+1)*x[i]-(i+1)*math.exp(-(i+1)*x[i]) for i in range(n)]
	return [-1/(n*2*math.sqrt(i)) for i in x]
	# return [-1/(n*i) for i in x]

tg='linear'
R=5
x0=np.array([1e-3]*n)
norm(x0)
Th0s=2*R**2

print('Algorithm 3')
print('====================================')
e=[1/4,1/9,1/16,1/25,1/36]
E=['1/4','1/9','1/16','1/25','1/36']
w=0
for eps in e:
	print('Epsilon =',E[w])
	x=x0
	k=0
	I=0
	S=0
	Sh=0
	Shx=0
	start_time=datetime.now()
	while(2*Th0s/eps**2>S+k-I):
		grad_g=Grad_g_s(x)
		if(grad_g[0]==None):
			grad_f=Grad_f(x)
			Ms=sum(np.power(grad_f,2)) # Square norm.
			h=eps/Ms
			y=np.array(np.subtract(x,np.multiply(h,grad_f)))
			norm(y)
			if(I==0 or f(X)<f(x)):
				X=x
			I+=1
			S+=1/Ms
			Sh+=h
			Shx+=h*x
		else:
			h=eps/M
			y=np.array(np.subtract(x,np.multiply(h,grad_g)))
			norm(y)
		x=y
		k+=1
	if(I!=0):
		X_mean=Shx/Sh
		print('Iterations =',k)
		# print('Productive steps =',I)
		print('X:',X)
		print('f(X):',f(X))
		print('X_mean:',X_mean)
		print('f(X_mean):',f(X_mean))
		t=[]
		for i in range(len(C)):
			t.append(G(C[i],b[i],X,'g'))
		print('Max G(X):',max(t))
	else:
		print("The set I is empty, k =",k)
	w+=1
	end_time=datetime.now()
	print('Time: {}'.format(end_time-start_time))
	print('------------------------------------')
