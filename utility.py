import numpy as np
from matplotlib import pyplot as plt
from pytriqs.gf.local import GfImFreq


def plot_giw(g, block = 'up', i = 0, j = 0, wi = 1025, wf = 1050, imag = True, **kwargs):
    x = np.array([iw.imag for iw in g.mesh])[wi:wf]
    y = g[block].data[wi:wf, i, j]
    if imag:
        plt.plot(x, y.imag, **kwargs)
    else:
        plt.plot(x, y.real, **kwargs)

def plot(x, y, **kwargs):
    plt.plot(x, y, **kwargs)

def save_plot(name, legend = False):
    if legend: plt.gca().legend()
    plt.savefig(name)
    plt.gca().cla()
    plt.gcf().clf()

def trace_giw(blockgf):
    """ gets BlockGf, returns GfImFreq"""
    tr_g = GfImFreq(beta = blockgf.beta, indices = [0], n_points = int(.5*len(blockgf.mesh)))
    tr_g.zero()
    norm = 0
    for s, b in blockgf:
        assert b.N1 == b.N2, "non-quadratic block"
        assert isinstance(b, GfImFreq), "need BlockGf of GfImFreq"
        for i in range(b.N1):
            tr_g += b[i, i]
            norm += 1
    tr_g /= norm
    return tr_g

def plot_bandstructure(bandstructure, **kwargs):
    imshowkwargs = {'extent': bandstructure.akw_extent,
                    'aspect': 'auto',
                    'origin': 'lower',
                    'vmin': 0,
                    'vmax': .3,
                    'interpolation': 'nearest'
    }
    imshowkwargs.update(kwargs)
    plt.imshow(bandstructure.akw.T, **imshowkwargs)
    plt.plot(bandstructure.k_enumeration, bandstructure.hk.real, color = 'white', alpha = .5)
    plt.gca().set_xticks(bandstructure.pathinds)
    plt.gca().set_xticklabels('$'+str(k)+'$' for k in bandstructure.path)
    plt.gca().set_xlabel('$k$')
    plt.gca().set_ylabel('$\omega$')
    plt.colorbar()

def trace_h(h):
    tr_h = 0
    norm = 0
    for s, b in h.items():
        for i in range(b.shape[0]):
            tr_h += b[i, i]
            norm += 1
    tr_h /= norm
    return tr_h
