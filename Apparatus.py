"""
Appartus to Algorithms.py

Eilam Levitov, July 2017
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from math import floor, sqrt, log 
import pywt
import scipy.sparse as sparse


# Signal Object
class Signal(object):
	def __init__(self, dim, N, n, sparsity):
		self.dim = dim									# Dimension of Signal [1,3] //fix
		self.N = N 										# Length of signal
		self.n = n 										# Nunber of samples
		self.sparsity = sparsity						# Sparsity value [0,1]
		self.pulses = int(self.N * self.sparsity)		# How many peaks in the generated signal
		self.generateSignal()							# Generate appropriate signal
# Generate the appropriate signal (limited to dim 1-3 right now)
	def generateSignal(self):
		if self.dim == 1:
			_x = self.generateRow()
		elif self.dim == 2: 
			_x = self.generateMatrix()
		elif self.dim == 3:
			_x = self.generateMatrix()
			for _ in range(self.dim - 1):
				np.append(_x,self.generateMatrix())
		self.signal = _x 								# Generated signal
	def generateMatrix(self):
		_x = np.zeros((self.N, self.N), dtype='complex')
		for i in range(len(_x[0])):
			_x[i] = self.generateRow()
		return _x
	def generateRow(self):
		pulseVals = [np.random.uniform(0, 1) for _ in range(self.pulses)]  ## Gaussian?
		_x = np.array(  pulseVals + [0] * (self.N-self.pulses), dtype='complex' )
		_x = _x[ np.random.permutation(self.N)-1]
		return _x


# Linear Operators Object
class Operator(object):
	def __init__(self, forward, adjoint, mask):
		self.forward = forward 							# Forward operator function
		self.adjoint = adjoint 							# Adjoint operator function
		self.mask = mask                                # Undersampling mask
		# self.dim = dim                                # Dimension of Signal [1,3] //fix
		# self.N = N                                    # Length of signal
		# self.n = n                                    # Nunber of samples
		# self.size = tuple([self.N for _ in range(self.dim)])
		# if idx == None:
		# 	self.idx = undersample(self.dim,self.N,self.n).astype(int)
# Forward operator
	def A(self, x):
		return self.mask * self.forward(x)
# Adjoint operator
	def AT(self, X):
		return self.adjoint(self.mask * X)
# Normal operator 
	def ATA(self,x):
		return self.adjoint(self.mask * self.forward(x))


# generate mask
def mask1d(N,n):
	idx = us_one(N,n)
	m = np.zeros(N)
	m[idx] = 1
	return m

# generate mask in 2d
def mask2d(N,n):
	m = np.zeros((N,N))
	for i in range(N):
		idx = us_one(N,n)
		m[i][idx] = 1
	return m

# Lipschitz's constant for l1 regularization
def L(A, AT=None, n=None):									###FIX dimension compatibility
	if callable(A) and callable(AT) and n != None:
		return np.real(2*max(np.linalg.eigvals(AT(A(np.identity(n))))))
	return np.real(2*max(np.linalg.eigvals(np.matrix.getH(A)@A)))

# Soft-Threshold
def SoftThreshReal(y, lamb):
	res = np.abs(y) - lamb
	xhat = res * np.sign(y)
	xhat[res < 0] = 0.
	return xhat
def SoftThreshComplex(y, lamb):
	res = np.abs(y)
	res = np.maximum(0,(res - lamb)) / (res + (res==0).astype(int))
	return y * res

# Derivative of soft-threshold
# 1 | (c<y ∧ y>=0) ∨ (y<0 ∧ y<-c)
# 0 | (otherwise)
def dST(y, lamb):
	res = np.abs(y) - lamb
	xhat = res
	xhat[res < 0] = 0
	xhat[res > 0] = 1
	return xhat

# Numerical Differentiation
def numericalDiff(f, x, lamb=None):
	dx = 1e-12
	f1 = f(x+dx,lamb)
	f2 = f(x-dx,lamb)
	return (f1 - f2) / (2 * dx)

# Stem plot on-line printing 
def printing(x,i,lamb):
	plt.clf()
	plt.stem(x)
	plt.title( 'Iteration {}, lambda {}'.format(i,lamb) )
	display.clear_output(wait=True)
	display.display(plt.gcf())

# Objective function evaluation (LASSO)
def F(A, x, b, lamb):
	f = 0.5 * (np.linalg.norm(A(x) - b, 2))**2
	g = lamb * np.linalg.norm(x, 1)
	return (f + g)
def F_matrix(A, x, b, lamb):
	f = 0.5 * (np.linalg.norm((A @ x) - b, 2))**2
	g = lamb * np.linalg.norm(x, 1)
	return (f + g)

# HW9-CS Apparatus
def imshowgray(im, vmin=None, vmax=None):
	plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
	
def wavMask(dims, scale):
	sx, sy = dims
	res = np.ones(dims)
	NM = np.round(np.log2(dims))
	for n in range(int(np.min(NM)-scale+2)//2):
		res[:int(np.round(2**(NM[0]-n))), :int(np.round(2**(NM[1]-n)))] = \
			res[:int(np.round(2**(NM[0]-n))), :int(np.round(2**(NM[1]-n)))]/2
	return res

def imshowWAV(Wim, scale=1):
	plt.imshow(np.abs(Wim)*wavMask(Wim.shape, scale), cmap = plt.get_cmap('gray'))
	
def coeffs2img(LL, coeffs):
	LH, HL, HH = coeffs
	return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

def unstack_coeffs(Wim):
		L1, L2  = np.hsplit(Wim, 2) 
		LL, HL = np.vsplit(L1, 2)
		LH, HH = np.vsplit(L2, 2)
		return LL, [LH, HL, HH]
	
def img2coeffs(Wim, levels=3):
	LL, c = unstack_coeffs(Wim)
	coeffs = [c]
	for i in range(levels-1):
		LL, c = unstack_coeffs(LL)
		coeffs.insert(0,c)
	coeffs.insert(0, LL)
	return coeffs  

def dwt2(im):
	coeffs = pywt.wavedec2(im, wavelet='db4', mode='per', level=3)
	Wim, rest = coeffs[0], coeffs[1:]
	for levels in rest:
		Wim = coeffs2img(Wim, levels)
	return Wim

def idwt2(Wim):
	coeffs = img2coeffs(Wim, levels=3)
	return pywt.waverec2(coeffs, wavelet='db4', mode='per')

def fft2c(x):
	return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(y):
	return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))


# Random permutation
def undersample(dim,N,n):
	if dim == 1:
		return(us_one(N, n))
	elif dim == 2:
		return(us_two(N,n))
	elif dim == 3:
		return(us_three(N,n))
def us_three(N,n):
	x = np.zeros((N,N,n))
	for j in range(N):
		x[i] = us_two(N,n)
	return x
def us_two(N,n):
	x = np.zeros((N,n))
	for i in range(N):
		x[i] = us_one(N,n)
	return x
def us_one(N,n):
	prm = np.random.permutation(N)
	return prm[:n]
