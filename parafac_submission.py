import time
import math
import igraph as ig
import scipy as sp
import numpy as np
from numpy import setdiff1d
from numpy import array, dot, ones, sqrt
from numpy.random import rand

from sktensor import sptensor, cp_als
from sktensor.core import nvecs, norm
from sktensor.ktensor import ktensor
from sktensor.sptensor import fromarray

import logging

logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger('CP')
_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-5
_DEF_FIT_METHOD = 'full'
_DEF_TYPE = np.float

__all__ = [
    'als',
    'opt',
    'wopt'
]


def _init(init, X, N, rank, dtype):
    """
    Initialization for CP models
    """
    Uinit = [None for _ in range(N)]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        for n in range(0, N):
            Uinit[n] = array(rand(X.shape[n], rank), dtype=dtype)
    elif init == 'nvecs':
        for n in range(0, N):
            Uinit[n] = array(nvecs(X, n, rank), dtype=dtype)
    else:
        raise 'Unknown option (init=%s)' % str(init)
    return Uinit


def cp_nmu(X, rank, **kwargs):
    # init options
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('max_iter', _DEF_MAXITER)
    fit_method = kwargs.pop('fit_method', _DEF_FIT_METHOD)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    epsilon = 1e-12
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))
    N = X.ndim
    normX = norm(X)
    U = _init(ainit, X, N, rank, dtype)
    fit = 0
    exectimes = []
    for itr in range(maxiter):
        tic = time.clock()
        fitold = fit
        for n in range(N):
            Y = ones((rank, rank), dtype=dtype)
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * dot(U[i].T, U[i])
            Y = dot(U[n], Y)
            Unew = U[n]
            tmp = X.uttkrp(U, n) + epsilon
            Unew = Unew * tmp
            Unew = Unew / (Y + epsilon)
            if itr == 0:
                lmbda = sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            U[n] = Unew / lmbda
        P = ktensor(U, lmbda)
        if fit_method == 'full':
            normresidual = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
            fit = 1 - (normresidual / normX ** 2)
        else:
            fit = itr
        fitchange = abs(fitold - fit)
        exectimes.append(time.clock() - tic)
        print("iter done")
        _log.debug(
            '[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' %
            (itr, fit, fitchange, exectimes[-1])
        )
        if itr > 0 and fitchange < conv:
            break
    return P, fit, itr, array(exectimes)


def own_single_concatenate(ten, other, axis):
    tshape = ten.shape
    oshape = other.shape
    if len(tshape) != len(oshape):
        raise ValueError("len(tshape) != len(oshape")
    oaxes = setdiff1d(range(len(tshape)), [axis])
    for i in oaxes:
        if tshape[i] != oshape[i]:
            raise ValueError("Dimensions must match")
    nsubs = [None for _ in range(len(tshape))]
    for i in oaxes:
        nsubs[i] = np.concatenate((ten.subs[i], other.subs[i]))
    nsubs[axis] = np.concatenate((
        ten.subs[axis], [x + tshape[axis] for x in other.subs[axis]]
    ))
    nsubs = tuple(nsubs)
    nvals = np.concatenate((ten.vals, other.vals))
    nshape = np.copy(tshape)
    nshape[axis] = tshape[axis] + oshape[axis]
    return sptensor(nsubs, nvals, nshape)


def add_dim(ten):
    tmp = ten
    tmp_list = list(tmp.subs)
    tmp_list.append([0]*len(tmp.subs[0]))
    tmp.subs = tuple(tmp_list)
    tmp_list = list(tmp.shape)
    tmp_list.append(1)
    tmp.shape = tuple(tmp_list)
    tmp.ndim = tmp.ndim +1
    tmp.vals = tmp.vals
    ten = tmp
    return(ten)



G = ig.Graph.Read_GraphML("dream_merge.graphml")

X = []
net_types = ["homology_score", "ppi1_score", "ppi2_score", "coexpr_score"]
for i,score in enumerate(net_types):
    G_DM = G.copy()
    score_nan = [math.isnan(e[score]) for e in G_DM.es]
    G_DM.es["score_nan"] = score_nan
    G_DM.delete_edges(G_DM.es.select(score_nan=True))
    G_DM_a = G_DM.get_adjacency()
    B = fromarray(sp.array(G_DM_a.data, dtype=float))
    G_DM_a = []
    X.append(B)

for x in X:
    add_dim(x)

cum = X[0]
for x in X[1:]:
    cum = own_single_concatenate(cum, x, axis=2)

R = 500
P, fit, itr, exectimes = cp_nmu(cum, R, init='random', conv=1e-6, max_iter=25)


A = P.U[0]
B = P.U[1]
#C = P.U[2]
labels_i = sp.zeros(cum.shape[0], dtype=np.int)
for i in range(cum.shape[0]):
    a = A[i,:]
    junk = sp.amax(a)
    idx = sp.argmax(a)
    A[i,:] = 0
    A[i,idx] = junk
    labels_i[i] = idx
    if sum(a) == 0:
        labels_i[i] = R


modules = [[] for i in range(R+1)]
for node,mod_idx in enumerate(labels_i):
    modules[mod_idx].append(G.vs[node]["id"])

#split into equal sized chunks if larger 100
included = []
for i,x in enumerate(x for x in modules[0:R]):
    if len(x)>=3:
        if len(x) <= 100:
            included.append(modules[i])
        else:
            tmp = np.array_split(x, np.ceil(len(x)/100.0))
            for t in tmp:
                included.append(t.tolist())


with open("./nonneg_parafac_submission.gmt", "w") as f:
    for i, m in enumerate(included):
        f.write("\t".join(['module_{}'.format(i), "1"] + [str(g) for g in m]) + "\n")