import numpy as np
import warnings
from scipy import stats
from copy import deepcopy

# include OptimizeResult class for machines on which scipy version is too old
class OptimizeResult(dict):
    """ Represents the optimization result.

    Parameters
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess, hess_inv : ndarray
        Values of objective function, Jacobian, Hessian or its inverse (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                                for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"



def minimizeSPSA_batches(func, init_params, batch_gen, dataset, param_stats={}, bounds=[-np.inf, np.inf], niter=100, paired=True,
                 a=1.0, c=1.0,
                 disp=False, callback=None):
    """
    Minimization of an objective function by a simultaneous perturbation
    stochastic approximation algorithm.

    Parameters
    ----------
    func: callable
        objective function to be minimized
    init_params: array-like
        starting point
    args: tuple
        extra arguments to be supplied to func
    bounds: array-like
        bounds on the variables
    scaling: array-like
        scaling by which to multiply step size and tolerances along different dimensions
    niter: int
        maximum number of iterations of the algorithm
    paired: boolean
        calculate gradient for same random seeds
    a: float
       algorithm scaling parameter for step size
    c: float
       algorithm scaling parameter for evaluation step size
    disp: boolean
        whether to output status updates during the optimization
    callback: callable
        called after each iteration, as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    scipy.optimize.OptimizeResult object
    """
    A = 0.01 * niter
    alpha = 0.602
    gamma = 0.101

    def reshape(x, nouns, verbs, params):

        n_nouns, n_verbs = len(nouns), len(verbs)
        noun_prop = n_nouns*param_stats["nouns"]
        xn = x[:noun_prop].reshape(n_nouns,param_stats["nouns"])
        xv = x[noun_prop:].reshape(n_verbs,param_stats["amb_verbs"])

        p_copy = deepcopy(params)
        for i, noun in enumerate(nouns):
            p_copy[noun]["p"] = xn[i].tolist()
        for i, verb in enumerate(verbs):
            p_copy[verb]["p"] = xv[i].tolist()

        return p_copy

    
    project = lambda x: np.clip(x, [bounds[0]]*len(x), [bounds[1]]*len(x))

    evo = []
    params = init_params
    
    min_loss = np.inf
    best_params = init_params

    for k in range(niter):
        loss = []
        for i in range(20):
            # extract batch and params
            batch = next(batch_gen)
            xl_nouns, xl_verbs = [], []
            l_nouns, l_verbs = [], []
            for point in batch:

                if params[point[1]]["learn"] and point[1] not in l_nouns:
                    l_nouns.append(point[1])
                    xl_nouns.append(params[point[1]]["p"])

                if params[point[0]]["learn"] and point[0] not in l_verbs:
                    l_verbs.append(point[0])
                    xl_verbs.append(params[point[0]]["p"])

            x = np.concatenate((np.array(xl_nouns).flatten(), np.array(xl_verbs).flatten()))
            
            ak = a/(k+1.0+A)**alpha
            ck = c/(k+1.0)**gamma
            delta = np.random.choice([-1, 1], size=len(x))
        
            # ensure evaluation points are feasible
            xplus = project(x + ck*delta)
            xminus = project(x - ck*delta)
            pplus = reshape(xplus, l_nouns, l_verbs, params)
            pminus = reshape(xminus, l_nouns, l_verbs, params)


            grad = (func(pplus, batch) - func(pminus, batch)) / (xplus-xminus)

            x = project(x - ak*grad)
            params = reshape(x, l_nouns, l_verbs, params)

        loss = func(params, dataset)
        if loss < min_loss:
            min_loss = loss
            best_params = params

        evo.append(loss)
        print("Ep {}: Mean loss:".format(k+1),loss)
    
    message = 'terminated after reaching max number of iterations'
    return OptimizeResult(fun=min_loss, x=best_params, nit=niter, nfev=2*niter, message=message, success=True, evo=evo)

class AverageBase(object):
    """
    Base class for averaged evaluation of noisy functions.
    """
    def __init__(self, N=30, paired=False):
        """
        Parameters
        ----------
        N: int
            number of calls to average over.
        paired: boolean
            if paired is chosen the same series of random seeds is used for different x
        """
        self._N = int(N)
        self.paired = paired
        if self.paired:
            self.uint32max = np.iinfo(np.uint32).max 
            self.seeds = list(np.random.randint(0, self.uint32max, size=int(N)))
        # cache previous iterations
        self.cache = {}
        # number of evaluations
        self.nev = 0

    @property
    def N(self):
        "number of evaluations"
        return self._N

    @N.setter
    def N(self, value):
        N = int(value)
        if self.paired and (N > self._N):
            Nadd = N - self._N
            self.seeds.extend(list(np.random.randint(0, self.uint32max, size=Nadd)))
        self._N = N

    def test0(self, x, type_='smaller', alpha=0.05, force=False, eps=1e-5, maxN=10000):
        """
        Compares the mean at x to zero.

        Parameters
        ----------
        type_: in ['smaller', 'equality']
            type of comparison to perform
        alpha: float
           significance level 
        force: boolean
            if true increase number of samples until equality rejected or meanse=eps or N > maxN
        eps: float 
        maxN: int
        """
        if force:
            while (self.test0(x, type_='equality', alpha=alpha, force=False, eps=eps)
                    and self(x)[1] > eps
                    and self.N < maxN):
                self.N *= 2.0

        mean, meanse = self(x)
        epscal = mean / meanse
        if type_ == 'smaller':
            return epscal < stats.norm.ppf(alpha)
        if type_ == 'equality':
            return np.abs(epscal) < stats.norm.ppf(1-alpha/2.0)
        raise NotImplementedError(type_)

class AveragedFunction(AverageBase):
    """Average of a function's return value over a number of runs.

        Caches previous results.
    """
    def __init__(self, func, fargs=None, **kwargs):
        """
        Parameters
        ----------
        func : callable
            function to average (called as `func(x, *fargs)`)
        fargs : tuple
            extra arguments for function
        """
        super(AveragedFunction, self).__init__(**kwargs)
        if fargs is not None:
            def funcf(x, **kwargs):
                return func(x, *fargs, **kwargs)
            self.func = funcf
        else:
            self.func = func

    def __call__(self, x):
        try:
            # convert to tuple (hashable!)
            xt = tuple(x)
        except TypeError:
            # if TypeError then likely floating point value
            xt = (x, )
        if xt in self.cache:
            Nold = len(self.cache[xt])
            if Nold < self.N:
                Nadd = self.N - Nold 
                if self.paired:
                    values = [self.func(x, seed=self.seeds[Nold+i]) for i in range(Nadd)]
                else:
                    values = [self.func(x) for i in range(Nadd)]
                self.cache[xt].extend(values)
                self.nev += Nadd
        else:
            if self.paired:
                values = [self.func(x, seed=self.seeds[i]) for i in range(self.N)]
            else:
                values = [self.func(x) for i in range(self.N)]
            self.cache[xt] = values 
            self.nev += self.N
        return np.mean(self.cache[xt]), np.std(self.cache[xt], ddof=1)/self.N**.5

    def diffse(self, x1, x2):
        """Standard error of the difference between the function values at x1 and x2""" 
        f1, f1se = self(x1)
        f2, f2se = self(x2)
        if self.paired:
            fx1 = np.array(self.cache[tuple(x1)])
            fx2 = np.array(self.cache[tuple(x2)])
            diffse = np.std(fx1-fx2, ddof=1)/self.N**.5 
            return diffse
        else:
            return (f1se**2 + f2se**2)**.5

    def test(self, xtest, x, type_='smaller', alpha=0.05):
        """
        Parameters
        ----------
        type_: in ['smaller', 'equality']
            type of comparison to perform
        alpha: float
            significance level
        """
        # call function to make sure it has been evaluated a sufficient number of times
        if type_ not in ['smaller', 'equality']:
            raise NotImplementedError(type_)
        ftest, ftestse = self(xtest)
        f, fse = self(x)
        # get function values
        fxtest = np.array(self.cache[tuple(xtest)])
        fx = np.array(self.cache[tuple(x)])
        if np.mean(fxtest-fx) == 0.0:
            if type_ == 'equality':
                return True
            if type_ == 'smaller':
                return False
        if self.paired:
            # if values are paired then test on distribution of differences
            statistic, pvalue = stats.ttest_rel(fxtest, fx, axis=None)
        else:
            statistic, pvalue = stats.ttest_ind(fxtest, fx, equal_var=False, axis=None)
        if type_ == 'smaller':
            # if paired then df=N-1, else df=N1+N2-2=2*N-2 
            df = self.N-1 if self.paired else 2*self.N-2
            pvalue = stats.t.cdf(statistic, df) 
            # return true if null hypothesis rejected
            return pvalue < alpha
        if type_ == 'equality':
            # return true if null hypothesis not rejected
            return pvalue > alpha

class DifferenceFunction(AverageBase):
    """Averages the difference of two function's return values over a number of runs
    """
    def __init__(self, func1, func2, fargs1=None, fargs2=None, **kwargs):
        """
        Parameters
        ----------
        func1,2 : callables
            functions to average (called as `func(x, *fargs)`)
        fargs1,2 : tuples
            extra arguments for functions
        kwargs: various
            accepts `AverageBase` kwargs and function kwargs
        """
        basekwargs = dict(N=kwargs.pop('N', 30),
                          paired=kwargs.pop('paired', False))
        super(DifferenceFunction, self).__init__(**basekwargs)
        if fargs1 is not None:
            def func1f(x, **kwargs):
                return func1(x, *fargs1, **kwargs)
        else:
            func1f = func1
        if fargs2 is not None:
            def func2f(x, **kwargs):
                return func2(x, *fargs2, **kwargs)
        else:
            func2f = func2
        self.funcs = [func1f, func2f]

    def __call__(self, x):
        try:
            # convert to tuple (hashable!)
            xt = tuple(x)
        except TypeError:
            # if TypeError then likely floating point value
            xt = (x, )
        for i, func in enumerate(self.funcs):
            ixt = i, xt
            if ixt in self.cache:
                Nold = len(self.cache[ixt])
                if Nold < self.N:
                    Nadd = self.N - Nold 
                    if self.paired:
                        values = [func(x, seed=self.seeds[Nold+i]) for i in range(Nadd)]
                    else:
                        values = [func(x) for i in range(Nadd)]
                    self.cache[ixt].extend(values)
                    self.nev += Nadd
            else:
                if self.paired:
                    values = [func(x, seed=self.seeds[i]) for i in range(self.N)]
                else:
                    values = [func(x) for i in range(self.N)]
                self.cache[ixt] = values 
                self.nev += self.N
        diff = np.asarray(self.cache[(0, xt)]) - np.asarray(self.cache[(1, xt)])
        return np.mean(diff), np.std(diff, ddof=1)/self.N**.5

class BisectException(Exception):
    pass

def bisect(func, a, b, xtol=1e-6, errorcontrol=True,
           testkwargs=dict(), outside='extrapolate',
           ascending=None,
           disp=False):
    """Find root by bysection search.

    If the function evaluation is noisy then use `errorcontrol=True` for adaptive
    sampling of the function during the bisection search.

    Parameters
    ----------
    func: callable
        Function of which the root should be found. If `errorcontrol=True`
        then the function should be derived from `AverageBase`.
    a, b: float
        initial interval
    xtol: float
        target tolerance for interval size
    errorcontrol: boolean
        if true, assume that function is instance of DifferenceFunction  
    testkwargs: only for `errorcontrol=True`
        see `AverageBase.test0`
    outside: ['extrapolate', 'raise']
        How to handle the case where f(a) and f(b) have same sign,
        i.e. where the root lies outside of the interval.
        If 'raise' throws a BisectException in this case.
    ascending: allow passing in directly whether function is ascending or not
        if ascending=True then it is assumed without check that f(a) < 0 and f(b) > 0
        if ascending=False then it is assumed without check that f(a) > 0 and f(b) < 0

    Returns
    -------
    float, root of function
    """
    search = True
    # check whether function is ascending or not
    if ascending is None:
        if errorcontrol:
            testkwargs.update(dict(type_='smaller', force=True))
            fa = func.test0(a, **testkwargs)
            fb = func.test0(b, **testkwargs)
        else:
            fa = func(a) < 0
            fb = func(b) < 0
        if fa and not fb:
            ascending = True
        elif fb and not fa:
            ascending =  False
        else:
            if disp:
                print('Warning: func(a) and func(b) do not have opposing signs -> no search done')
            if outside == 'raise':
                raise BisectException()
            search = False

    # refine interval until it has reached size xtol, except if root outside
    while (b-a > xtol) and search:
        mid = (a+b)/2.0
        if ascending:
            if ((not errorcontrol) and (func(mid) < 0)) or \
                    (errorcontrol and func.test0(mid, **testkwargs)):
                a = mid 
            else:
                b = mid
        else:
            if ((not errorcontrol) and (func(mid) < 0)) or \
                    (errorcontrol and func.test0(mid, **testkwargs)):
                b = mid 
            else:
                a = mid
        if disp:
            print('bisect bounds', a, b)
    # interpolate linearly to get zero
    if errorcontrol:
        ya, yb = func(a)[0], func(b)[0]
    else:
        ya, yb = func(a), func(b)
    m = (yb-ya) / (b-a)
    res = a-ya/m
    if disp:
        print('bisect final value', res)
    return res

class _memoized(object):
    """Decorator. Caches a function's return value each time it is called.

    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    Can be turned of by passing `memoize=False` when calling the function.

    If the function arguments are not hashable, then no caching is attempted
    and the function is evaluated at every call. An error can instead be raised
    by passing `nothashable=raise` when calling the function.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.nev = 0

    def __call__(self, *args, **kwargs):
        # if args is not Hashable we can't cache
        # easier to ask for forgiveness than permission
        memoize = kwargs.pop('memoize', True)
        nothashable = kwargs.pop('nothashable', 'ignore')
        if memoize:
            try:
                index = ()
                for arg in args:
                    # unwrap iterable arguments
                    # (needed e.g. for np.ndarray which is not hashable)
                    try:
                        index += tuple(arg)
                    except TypeError:
                        index += (arg, ) 
                # try to also recompute if kwargs changed
                for item in kwargs.values():
                    try:
                        index += (float(item), )
                    except:
                        pass
                if index in self.cache:
                    return self.cache[index]
                else:
                    value = self.func(*args, **kwargs)
                    self.nev += 1
                    self.cache[index] = value
                    return value
            except TypeError:
                if nothashable == 'raise':
                    raise TypeError('Not hashable: %s' % str(args))
                else:
                    print('not hashable', args)
                    self.nev += 1
                    return self.func(*args, **kwargs)
        else:
            self.nev += 1
            return self.func(*args, **kwargs)
