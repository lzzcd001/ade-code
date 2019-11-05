import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def rings_sample(N, D, sigma=0.1, radia=np.array([1, 3])):
    assert D >= 2
    
    angles = np.random.rand(N) * 2 * np.pi
    noise = np.random.randn(N) * sigma
    
    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    radia_inds = np.random.choice(len(radia), N, p=weights)
    radius_samples = radia[radia_inds] + noise
    
    xs = (radius_samples) * np.sin(angles)
    ys = (radius_samples) * np.cos(angles)
    X = np.vstack((xs, ys)).T.reshape(N, 2)
    
    result = np.zeros((N, D))
    result[:, :2] = X
    if D > 2:
        result[:, 2:] = np.random.randn(N, D - 2) * sigma
    return result

def rings_log_pdf_grad(X, sigma=0.1, radia=np.array([1, 3])):
    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    norms = np.linalg.norm(X[:, :2], axis=1)

    result = np.zeros(np.shape(X))

    grads = []
    for i in range(len(X)):
        log_pdf_components = -0.5 * (norms[i] - radia) ** 2 / (sigma ** 2)
        log_pdf = logsumexp(log_pdf_components + np.log(weights))
        neg_log_neg_ratios = log_pdf_components - log_pdf

        gs_inner = np.zeros((len(radia), 1))
        for k in range(len(gs_inner)):
            gs_inner[k] = -(norms[i] - radia[k]) / (sigma ** 2)

        grad_1d = np.dot(gs_inner.T, np.exp(neg_log_neg_ratios + np.log(weights)))
        angle = np.arctan2(X[i, 1], X[i, 0])
        grad_2d = np.array([np.cos(angle), np.sin(angle)]) * grad_1d
        grads += [grad_2d]
    
    result[:, :2] = np.array(grads)
    if X.shape[1] > 2:
        # standard normal log pdf gradient
        result[:, 2:] = -X[:, 2:] / (sigma ** 2)
    
    return result

def rings_log_pdf(X, sigma=0.1, radia=np.array([1, 3])):

    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    norms = np.linalg.norm(X[:, :2], axis=1)

    result = np.zeros(np.shape(X)[0])

    for i in range(len(X)):
        log_pdf_components = -0.5 * (norms[i] - radia) ** 2 / (sigma ** 2) - \
                              0.5 * np.log(2*np.pi*sigma**2) - \
                              np.log(2*np.pi * radia)
        result[i] = logsumexp(log_pdf_components + np.log(weights))
    
    if X.shape[1] > 2:
        # stand+rd normal log pdf gradient
        result += np.sum(-0.5*np.log(2*np.pi*sigma**2) -0.5 * (X[:, 2:]**2) / (sigma ** 2),1)
    
    return result

def apply_whiten(data):
    
    mean = data.mean(0)
    data = data - data.mean(0)
    u, s, vt = np.linalg.svd(data[:10**4])
    W = vt.T/s * np.sqrt(u.shape[0])
    data = np.dot(data, W)
    return data, W, mean

def inv_whiten(data, W, mean):
    return data.dot(np.linalg.inv(W)) + mean

def apply_scale(data):
    
    mean = data.mean(0)
    data = data - data.mean(0)
    std  = data.std(0)
    data /= std

    return data, std, mean

def inv_scale(data, std, mean):
    return data*std + mean

def apply_itanh(data):
    
    m  =  data.min(axis=0)
    data -= m
    ptp = data.ptp(axis=0) 
    data /= ptp / 0.98 / 2
    data -= 0.98

    data = np.arctanh(data)

    m2 = data.mean(0)
    data -= m2

    return data, ptp, m, m2

def inv_itanh(data, ptp, m, m2):

    data += m2
    data = np.tanh(data)
    data += 0.98
    data *= ptp / 0.98/2
    data += m

    return data


def support_1d(fun, x):
    assert 1<=x.ndim<=2
    return fun(x) if x.ndim == 2 else fun(x[None,:])[0]


class Dataset(object):

    def sample(self, n):
        raise NotImplementedError

    def sample_two(self, n1, n2):
        raise NotImplementedError


class ToyDataset(Dataset):

    def sample(self, n):
        raise NotImplementedError

    def sample_two(self, n1, n2):
        return self.sample(n1), self.sample(n2)

    def logpdf_multiple(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        return support_1d(self.logpdf_multiple, x)

    def log_pdf(self, x):
        return support_1d(self.logpdf_multiple, x)

    def log_pdf_multile(self, x):
        return self.logpdf_multiple(x)


    def dlogpdf(self, x):
        return grad_multiple(x)

    def grad_multiple(self, x):
        raise NotImplementedError

    def grad(self, x):
        return self.grad_multiple(self.logpdf, x)

    def score(self, x):
        return -0.5*self.grad_multiple(x**2,1)

class Spiral(ToyDataset):
    
    def __init__(self, sigma=0.5, D = 2, eps=1, r_scale=1.5, starts=np.array([0.0,2.0/3,4.0/3]) * np.pi, 
                length=np.pi):

        self.sigma = sigma
        self.L= length
        self.r_scale = r_scale
        self.D = D
        self.eps = eps # add a small noise at the center of spiral
        self.starts= starts
        self.nstart= len(starts)
        self.name = "spiral"
        self.has_grad = False

    def _branch_params(self, a, start):
        
        n = len(a)
        a = self.L * ((a)**(1.0/self.eps))+ start
        r = (a-start)*self.r_scale
        
        m = np.zeros((n, self.D))
        s = np.ones((n, self.D)) * self.sigma
        
        m[:,0] = r * np.cos(a)
        m[:,1] = r * np.sin(a)
        s[:,:2] = (a[:,None]-start)/self.L * self.sigma + 0.1

        return m, s

    def _branch_params_one(self, a, start):
        
        a = self.L * ((a)**(1.0/self.eps))+ start
        r = (a-start)*self.r_scale
        
        m = np.zeros((self.D))
        s = np.ones((self.D)) * self.sigma
        
        m[0] = r * np.cos(a)
        m[1] = r * np.sin(a)
        s[:2] = (a-start)/self.L * self.sigma

        return m, s

    def sample(self, n):
        
        data = np.zeros((n+self.nstart, self.D))
        batch_size = np.floor_divide(n+self.nstart,self.nstart)
        
        for si, s in enumerate(self.starts):
            m = np.floor_divide(n,self.nstart)
            data[si*batch_size:(si+1)*batch_size] = self.sample_branch(batch_size, s)
        return  data[:n,:]

        
        
    def sample_branch(self, n, start):
        
        a = np.random.uniform(0,1,n)

        m, s = self._branch_params(a, start) 

        data = m + np.random.randn(n, self.D) * s
        return data

    def _conditional_pdf(self, a, x):
        
        n = x.shape[0]
        p = np.array((n,self.nstart))

        for si, s in enumerate(self.starts):
            
            m, s = self._branch_params(a, s)
            pdf[:,si] = norm.logpdf(x, loc = m, scale = s).sum(1)
            pdf[:,si] -= np.log(self.nstart)

        return np.sum(np.exp(pdf), 1)

    def _conditional_pdf_one(self, a, x):
        
        pdf = np.zeros((self.nstart))

        for si, s in enumerate(self.starts):
            
            m, s = self._branch_params_one(a, s)
            pdf[si] = norm.logpdf(x, loc = m, scale = s).sum()
            pdf[si] -= np.log(self.nstart)

        return np.sum(np.exp(pdf))

    def _conditional_dpdf_one_dim(self, a, x, D):

        dpdf = np.zeros((self.nstart))
        
        for si, s in enumerate(self.starts):
            
            m, s = self._branch_params_one(a, s)
            dpdf[si] = np.exp(norm.logpdf(x, loc = m, scale = s).sum()) * ( - x[D] + m[D]) / s[D]**2
            dpdf[si] /= self.nstart

        return dpdf.sum()

    def pdf_one(self, x, *args, **kwargs):
        
        return quad(self._conditional_pdf_one, 0, 1, x, *args, **kwargs)[0]

    def dpdf_one(self, x, *args, **kwargs):
        
        dpdf = np.zeros(self.D)
        for d in range(self.D):
            dpdf[d] = quad(self._conditional_dpdf_one_dim, 0, 1, (x, d), *args, **kwargs)[0]
        return dpdf

    def grad_one(self, x, *args, **kwargs):
        
        return self.dpdf_one(x, *args, **kwargs) / self.pdf_one(x, *args, **kwargs)


class Funnel(ToyDataset):
    
    def __init__(self, sigma=2.0, D=2, lim=10.0):
    
        self.sigma=sigma
        self.D=D
        self.lim=lim
        self.low_lim = 0.000
        self.thresh   = lambda x: np.clip(np.exp(-x), self.low_lim, self.lim)
        self.name="funnel"
        self.has_grad = True
        
        
    def sample(self, n):
        
        data = np.random.randn(n, self.D)
        data[:,0]  *= self.sigma
        v =  self.thresh(data[:,0:1])
        data[:,1:] = data[:,1:] * np.sqrt(v)
        return data
    
    def grad_multiple(self, x):
        
        N = x.shape[0]
        grad = np.zeros((N, self.D))
        x1 = x[:,0]
        
        v = np.exp(-x1)
        
        dv  = -1*v
        dlv = -np.ones_like(v)
        
        dv[(v) < self.low_lim] = 0
        dv[(v) > self.lim] = 0
        
        dlv[(v) < self.low_lim] = 0
        dlv[(v) > self.lim] = 0
        
        grad[:,0] = -x1/self.sigma**2 - (self.D-1)/2.0 * dlv - 0.5*(x[:,1:]**2).sum(1) * (-dv)/v/v
        grad[:,1:]= - x[:,1:] / self.thresh(x1)[:,None]
        return grad
    
    def logpdf_multiple(self, x):
        v = self.thresh(x[:,0])
        return norm.logpdf(x[:,0], 0, self.sigma) + norm.logpdf(x[:,1:], 0, np.sqrt(v)[:,None]).sum(1)

class Ring(ToyDataset):

    def __init__(self, sigma=0.2, D=2, nring = 1):

        assert D >= 2
        
        self.sigma = sigma
        self.D = D
        
        self.radia = np.array([5])
        self.name  = "ring"
        self.has_grad = True
        
    def grad_multiple(self, X):
        return rings_log_pdf_grad(X, self.sigma, self.radia)

    def logpdf_multiple(self, X):
        return rings_log_pdf(X, self.sigma, self.radia)

    def sample(self, N):
        samples = rings_sample(N, self.D, self.sigma, self.radia)
        return samples

class Multiring(ToyDataset):

    def __init__(self, sigma=0.2, D=2):

        assert D >= 2
        
        self.sigma = sigma
        self.D = D
        self.radia = np.array([1, 3, 5])
        self.name  = "multiring"
        self.has_grad = True
        
    def grad_multiple(self, X):
        return rings_log_pdf_grad(X, self.sigma, self.radia)

    def logpdf_multiple(self, X):
        return rings_log_pdf(X, self.sigma, self.radia)

    def sample(self, N):
        samples = rings_sample(N, self.D, self.sigma, self.radia)
        return samples

class Cosine(ToyDataset):

    def __init__(self, D=2, sigma=1.0, xlim = 4, omega=2, A=3):

        assert D >= 2
        
        self.sigma = sigma
        self.xlim  = xlim
        self.A = A
        self.D = D
        self.name="cosine"
        self.has_grad=True
        self.omega = omega
        
    def grad_multiple(self, X):

        grad = np.zeros_like(X)
        x0= X[:,0]
        m = self.A*np.cos(self.omega*x0)
        #grad[:,0] = -X[:,0]/self.xlim**2 - (X[:,1]-m)/self.sigma**2 * self.A * np.sin(self.omega*X[:,0])*self.omega
        grad[:,0] = -(X[:,1]-m)/self.sigma**2 * self.A * np.sin(self.omega*X[:,0])*self.omega
        grad[:,1] = -(X[:,1]-m)/self.sigma**2
        grad[np.abs(x0)>self.xlim,:]=0
        if self.D>2:
            grad[:,2:] = -X[:, 2:]
        
        return grad

    def logpdf_multiple(self, X):
        
        x0 = X[:,0]
        x1_mu = self.A * np.cos(self.omega*x0)
        
        #logpdf = norm.logpdf(x0, 0, self.xlim)
        logpdf  = -np.ones(X.shape[0])*np.log(2*self.xlim)
        logpdf += norm.logpdf(X[:,1], x1_mu, self.sigma)
        logpdf += np.sum(norm.logpdf(X[:,2:], 0, 1), -1)

        logpdf[np.abs(x0)>self.xlim] = -np.inf
        
        return logpdf

    def sample(self, N):
        x0 = np.random.uniform(-self.xlim, self.xlim, N)
        x1 = self.A * np.cos(self.omega*x0)

        x = np.random.randn(N, self.D)
        x[:,0] = x0
        x[:,1] *= self.sigma
        x[:,1] += x1
        return x

class Uniform(ToyDataset):

     def __init__(self, D=2,lims = 3):
         self.lims = lims
         self.D = D
         self.has_grad = True

     def sample(self,n):
         return 2*(np.random.rand(n, self.D) - 0.5) * self.lims

     def logpdf_multiple(self, x):
         pdf = - np.ones(x.shape[0]) * np.inf
         inbounds = np.all( (x<self.lims) * ( x > -self.lims), -1)
         pdf[inbounds] = -np.log((2*self.lims)**self.D)
         return pdf
         
     def grad_multiple(self, x):
         
         return np.zeros_like(x) 

class Banana(ToyDataset):
    
    def __init__(self, bananicity = 0.2, sigma=2, D=2):
        self.bananicity = bananicity
        self.sigma = sigma
        self.D = D
        self.name = "banana"
        self.has_grad = True

    def logpdf_multiple(self,x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.D
        logp =  norm.logpdf(x[:,0], 0, self.sigma) + \
                norm.logpdf(x[:,1], self.bananicity * (x[:,0]**2-self.sigma**2), 1)
        if self.D > 2:
            logp += norm.logpdf(x[:,2:], 0,1).sum(1)

        return logp

    def sample(self, n):
        
        X = np.random.randn(n, self.D)
        X[:, 0] = self.sigma * X[:, 0]
        X[:, 1] = X[:, 1] + self.bananicity * (X[:, 0] ** 2 - self.sigma**2)
        if self.D > 2:
            X[:,2:] = np.random.randn(n, self.D - 2)
        
        return X

    def grad_multiple(self, x):
        
        x = np.atleast_2d(x)
        assert x.shape[1] == self.D

        grad = np.zeros(x.shape)
        
        quad = x[:,1] - self.bananicity * (x[:,0]**2 - self.sigma**2)
        grad[:,0] = -x[:,0]/self.sigma**2 + quad * 2 * self.bananicity * x[:,0]
        grad[:,1] = -quad
        if self.D > 2:
            grad[:,2:] = -x[:, 2:]
        return grad

