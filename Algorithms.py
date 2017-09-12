"""
Linear Inverse Problems 
Applicable Optimization
		Algorithms		

Eilam Levitov, July 2017

with support from 
Jon Tamir - el Chapo del MikGroup
"""

import numpy as np
import Apparatus as ap
from math import sqrt, log
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator

## Iterative Shrinkage Thresholding 
class ISTA(object):
	def __init__(self, kwargs):
		self.x_init = kwargs.get('x_init',None)				# Initial Estimate
		self.niter = kwargs.get('niter',None)				# Number of Iterations
		self.A = kwargs.get('A',None)						# Forward operator
		self.AT = kwargs.get('AT',None)						# Adjoint
		self.ATA = kwargs.get('ATA',None)					# Normal
		self.y = kwargs.get('y',None)						# Observation
		self.Reg = kwargs.get('Reg',None)					# Regulator
		self.lamb = kwargs.get('lamb',None)					# Lambda (Thresholding) parameter
		self.signal = kwargs.get('signal',None)				# Signal Object // Can be done without
		self.N = kwargs.get('N',None)						# Length of signal
		self.x_true = kwargs.get('signal',None)	 			# True Signal
		self.process()
	def process(self):
		ATy = self.AT(self.y)
		if self.x_init != None:
			_x = self.x_init
		else:
			_x = ATy.copy() 
		l = 0.5/(ap.L(self.A,self.AT,self.N))
		self.conv = np.zeros(self.niter + 1)			# To plot convergence 
		self.conv[0] = np.linalg.norm(_x - self.x_true, 2)
		self.objective = np.zeros(self.niter + 1)
		self.objective[0] = ap.F(self.A, _x, self.y, self.lamb)
		self.counter = self.niter
		for i in range(self.niter):
			gamma_ista = _x - 2 * l * ( self.ATA(_x) - ATy  )
			_x = self.Reg( gamma_ista, self.lamb * l)
			self.conv[i + 1] = np.linalg.norm(_x - self.x_true, 2)
			self.objective[i + 1] = ap.F(self.A, _x, self.y, self.lamb)
		self.result = _x 									# Reconstructed signal


## Fast Iterative Shrinkage Thresholding 
class FISTA(object):
	def __init__(self, kwargs):
		self.x_init = kwargs.get('x_init',None)				# Initial Estimate
		self.niter = kwargs.get('niter',None)				# Number of Iterations
		self.A = kwargs.get('A',None)						# Forward operator
		self.AT = kwargs.get('AT',None)						# Adjoint
		self.ATA = kwargs.get('ATA',None)					# Normal
		self.y = kwargs.get('y',None)						# Observation
		self.Reg = kwargs.get('Reg',None)					# Regulator
		self.lamb = kwargs.get('lamb',None)					# Lambda (Thresholding) parameter
		self.N = kwargs.get('N',None)						# Length of signal
		self.x_true = kwargs.get('signal',None)	 			# True Signal
		self.process()
	def process(self):
		ATy = self.AT(self.y)
		if self.x_init != None:
			_x = self.x_init
		else:
			_x = ATy.copy()
		l = 1/(ap.L(self.A,self.AT,self.N))
		x_new = _x
		t_new = 1
		self.conv = np.zeros(self.niter + 1)			# To plot convergence 
		self.conv[0] = np.linalg.norm(x_new - self.x_true, 2)
		self.objective = np.zeros(self.niter + 1)
		self.objective[0] = ap.F(self.A, x_new, self.y, self.lamb)				
		self.counter = self.niter
		for i in range(self.niter):
			x_old = x_new
			gamma_fista = _x - 2 * l * ( self.ATA(_x) - ATy)  
			x_new = self.Reg( gamma_fista, self.lamb * l ) ## NaN? 
			t_old = t_new
			t_new = ( ( 1 + sqrt(1 + 4 * t_old * t_old) ) / 2 )
			_x = x_new + ( (t_old - 1)/(t_new) ) * ( x_new - x_old )
			self.conv[i + 1] = np.linalg.norm(_x - self.x_true, 2)
			self.objective[i + 1] = ap.F(self.A, _x, self.y, self.lamb)
		self.result = _x									# Reconstructed signal


## Alternating Direction Method of Multipliers
class ADMM(object):
	def __init__(self, kwargs):
		self.x_init = kwargs.get('x_init',None)				# Initial Estimate
		self.niter = kwargs.get('niter',None)				# Number of Iterations
		self.A = kwargs.get('A',None)						# Forward operator
		self.AT = kwargs.get('AT',None)						# Adjoint
		self.ATA = kwargs.get('ATA',None)					# Normal
		self.y = kwargs.get('y',None)						# Observation
		self.Reg = kwargs.get('Reg',None)					# Regulator
		self.lamb = kwargs.get('lamb',None)					# Lambda (Thresholding) parameter
		self.N = kwargs.get('N',None)						# Length of signal
		self.x_true = kwargs.get('signal',None)	 			# True Signal
		self.process()
	def process(self):
			maximum = np.maximum
			absolute = np.absolute
			sign = np.sign
			ATy = self.AT(self.y)
			if self.x_init != None:
				_x = self.x_init
			else:
				_x = ATy.copy()
			self.dim = _x.ndim
			self.shape = _x.shape 							
			_x = _x.ravel()
			ATy = ATy.ravel()
			_z = np.zeros(_x.shape)
			_u = np.zeros(_x.shape)
			self.rho = 1/(ap.L(self.A,self.AT,self.N)) 		# rho // Arbitrary(?) - fix
			lamb_over_rho =  self.lamb / self.rho
			dimension = tuple([self.N**self.dim for _ in range(2)])
			_A = LinearOperator(dimension, matvec=self.mv, rmatvec=self.mv)
			self.conv = np.zeros(self.niter + 1)		# To plot convergence 
			self.conv[0] = np.linalg.norm(_x - self.x_true.ravel(), 2)
			self.objective = np.zeros(self.niter + 1)
			self.objective[0] = ap.F(self.A, _x.reshape(self.shape), self.y, self.lamb)
			self.counter = self.niter
			for i in range(self.niter):
				gamma_admm = (ATy + self.rho * (_z - _u)).ravel()
				_x =  lsq_linear(_A, gamma_admm, verbose=0, max_iter=100, tol=1e-14, lsmr_tol=1e-14)['x']
				_u = _x + _u
				_z = self.Reg(_u, lamb_over_rho)
				_u = _u - _z
				self.conv[i + 1] = np.linalg.norm(_x - self.x_true.ravel(), 2)
				self.objective[i + 1] = ap.F(self.A, _x.reshape(self.shape), self.y, self.lamb)
			_x.shape = self.shape
			self.result = _x								# Reconstructed signal
			
	def mv(self,x): 										# Reshaping for LinearOperator() 
		x = x.reshape(self.shape)
		return (self.ATA(x) + self.rho * x).ravel()


## Alternating Minimization
class AM(object):
	def __init__(self, kwargs):
		self.x_init = kwargs.get('x_init',None)				# Initial Estimate
		self.niter = kwargs.get('niter',None)				# Number of Iterations
		self.A = kwargs.get('A',None)						# Forward operator
		self.AT = kwargs.get('AT',None)						# Adjoint
		self.ATA = kwargs.get('ATA',None)					# Normal
		self.y = kwargs.get('y',None)						# Observation
		self.Reg = kwargs.get('Reg',None)					# Regulator
		self.lamb = kwargs.get('lamb',None)					# Lambda (Thresholding) parameter
		self.N = kwargs.get('N',None)						# Length of signal
		self.x_true = kwargs.get('signal',None)	 			# True Signal
		self.t_m = kwargs.get('t_m',None)
		self.t_p = kwargs.get('t_p',None)
		self.process()
	def process(self):
		ATy = self.AT(self.y)
		# t_m = 1
		# t_p = 10
		if self.x_init != None:
			_x = self.x_init
		else:
			_x = ATy.copy()
		m = np.abs(_x)
		p = np.angle(_x)
		shifts = np.random.uniform(-np.pi,np.pi,size=10)
		# l = 1/(ap.L(self.A,self.AT,self.N))				# Lipschitz constant
		self.conv = np.zeros(self.niter + 1)				# To plot convergence 
		self.conv[0] = np.linalg.norm(_x - self.x_true, 2)
		self.objective = np.zeros(self.niter + 1)
		self.objective[0] = ap.F(self.A, _x, self.y, self.lamb)
		k = 0
		K = 10
		h = 1
		for i in range(self.niter):
			expP = np.exp(1j * p)
			gamma_am = ATy - self.ATA(m * expP) 	# Residual
			m = self.Reg(m + self.t_m * np.real(np.conj(expP) * gamma_am), self.lamb*self.t_m) # mag. update
			pShift = shifts[np.random.randint(0,5)]	# random phase shift 
			p = self.Reg(p + self.t_p * np.real((-1j) * np.conj(m) * np.conj(expP) * gamma_am + pShift), self.lamb*self.t_p) - pShift  # phase update (update expP before?)
			_x = m*np.exp(1j*p)			
			self.conv[i + 1] = np.linalg.norm(_x - self.x_true, 2)
			self.objective[i + 1] = ap.F(self.A, _x, self.y, self.lamb)
			t_p = 1 / (np.max(np.square(m)) + 1e-14) * h

			k = k + 1
			if k == K:
				k = 0 
				K = K * 2
				h = h / 2
				t_m = t_m * h
		self.result = _x									# Reconstructed signal
"""
		fprintf(' %d: Mag prox\n', it);
    args.name = sprintf('%s-mag', mpargs.name);
    m = Pm(m + alpham * real(M.T * (conj(expPp) .* r )), alpham, args)
    fprintf(' %d: Done\n', it);

    fprintf(' %d: Phase prox\n', it);
    args.name = sprintf('%s-phs', mpargs.name);
    p = Pp(p + alphap * real(-1j * (P.T * (conj(Mm) .* conj(expPp) .* r)))...
        + w, alphap, args) - w;
    fprintf(' %d: Done\n', it);
"""
## Approximate Message Passing 
class AMP(object):
	def __init__(self, kwargs):
		self.x_init = kwargs.get('x_init',None)				# Initial Estimate
		self.niter = kwargs.get('niter',None)				# Number of Iterations
		self.A = kwargs.get('A',None)						# Forward operator
		self.AT = kwargs.get('AT',None)						# Adjoint
		self.ATA = kwargs.get('ATA',None)					# Normal
		self.y = kwargs.get('y',None)						# Observation
		self.Reg = kwargs.get('Reg',None)					# Regulator
		self.lamb = kwargs.get('lamb',None)					# Lambda (Thresholding) parameter
		self.N = kwargs.get('N',None)						# Length of signal
		self.x_true = kwargs.get('signal',None)	 			# True Signal
		self.process()
	def process(self):
		ATy = self.AT(self.y)
		if self.x_init != None:
			_x = self.x_init
		else:
			_x = ATy.copy()
		_z = y.copy()
		weight = (1/A.shape[0]) # 1 / n 
		self.conv = np.zeros(self.niter + 1)			# To plot convergence 
		self.conv[0] = np.linalg.norm(_x - self.x_true, 2)
		self.objective = np.zeros(self.niter + 1)
		self.objective[0] = ap.F(self.A, _x, self.y, self.lamb)
		for i in range(self.niter):
			gamma_amp = _x + self.AT(_z) 				# Residual
#			sortedGamma = sorted(abs(gamma_amp), reverse=True)
#	      	lamb = sortedGamma[cutoffValue] 			# FIX Cutoff Value
			_x = self.Reg(gamma_amp, lamb)
			_z = (y - self.A(_x)) + _z * weight * sum(ap.dST(gamma_amp,lamb))	
			self.conv[i + 1] = np.linalg.norm(_x - self.x_true, 2)
			self.objective[i + 1] = ap.F(self.A, _x, self.y, self.lamb)
		self.result = _x	




	##################
	## Relic Bunker ##
	##################

		# self.useError = kwargs.get('error',None)			# Bolean to dictate iteration rule
		# self.error = kwargs.get('error',None)				# Required error if useError = True
		# self.signal = kwargs.get('signal',None)			# Signal
		# self.n = kwargs.get('n',None)						# Number of samples
		# self.signalObject = kwargs.get('signalObject',None)	# Signal Object // Can be done without
		# self.n = self.signalObject.n						# Number of samples
		# self.N = self.signalObject.N						# Length of signal
		# self.dim = self.signalObject.dim 					# Signal dimension
		# self.x_true = self.signalObject.signal			# True Signal

		# if self.useError:									# Error limited iteration
		# 	self.counter = 0
		# 	self.err = np.linalg.norm(x_i - self.x_true, 2)
		# 	while self.err > self.error:
		# 		self.counter += 1
		# 		gamma_ista = x_i - 2 * l *  ( self.ATA(x_i) - ATy )
		# 		x_i = np.nan_to_num(self.Reg( gamma_ista, self.lamb * l ))
		# 		if (self.counter % 10) == 0:
		# 			self.conv[i//10 + 1] = np.linalg.norm(x_i - self.x_true, 2)
		# 			self.objective[i//10 + 1] = ap.F(self.A, x_i, self.y, self.lamb)
		# else:												# Numbered iteration