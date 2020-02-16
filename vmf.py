# vmf sampler tests
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

def add_norm_noise(munorm, eps, max_norm):
    trand = np.random.uniform(low=0, high=1)*eps
    return np.clip(munorm, 0, max_norm-eps) + trand

def sample_vMF(mu, kappa, norm_eps, max_norm):
    id_dim = len(mu)
    munorm  = np.linalg.norm(mu)
    munoise = add_norm_noise(munorm,norm_eps, max_norm)
    if munorm > 1e-10:
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, id_dim)
        wtorch = w*np.ones(id_dim)
        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu/munorm, id_dim)
        # compute new point
        scale_factr = np.sqrt(np.ones(id_dim) - np.power(wtorch,2))
        orth_term = v * scale_factr
        muscale = mu * wtorch / munorm
        sampled_vec = (orth_term + muscale)*munoise
    else:
        rand_draw = np.random.randn(id_dim)
        rand_draw = rand_draw / np.linalg.norm(rand_draw)
        rand_norms = (np.random.rand(1) * norm_eps)
        sampled_vec = rand_draw*rand_norms
    return sampled_vec

def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa) # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

    while True:
        z = np.random.beta(dim / 2., dim / 2.)  #concentrates towards 0.5 as d-> inf
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u): #thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
            return w

def _sample_orthonormal_to(mu, dim):
    """Sample point on sphere orthogonal to mu.
    """
    v = np.random.randn(dim)
    rescale_value = np.dot(mu,v)
    proj_mu_v = mu * rescale_value
    ortho = v - proj_mu_v
    ortho_norm = np.linalg.norm(ortho)
    return ortho / ortho_norm

def main():
    arr = []
    x = np.zeros(100)
    y = np.zeros(100)
    for i in range(100):
        v = sample_vMF(array([2.0,2.0]), 15.0, 0.1, 10.0)
        x[i] = v[0]
        y[i] = v[1]
    plt.scatter(x, y)
    plt.savefig('python_vmf.png')

if __name__== "__main__":
    main()
