import numpy as np
from numbers import Number


def _split_rng(rng, N):
    keyz = []
    for i in range(N):
        rng, subkey = jax.random.split(rng)
        keyz.append(rng)
    return keyz

def rfftnfreq_2d(shape, spacing, dtype=np.float64):
    """Broadcastable "``sparse``" wavevectors for ``numpy.fft.rfftn``.

    Parameters
    ----------
    shape : tuple of int
        Shape of ``rfftn`` input.
    spacing : float or None, optional
        Grid spacing. None is equivalent to a 2π spacing, with a wavevector period of 1.
    dtype : dtype_like

    Returns
    -------
    kvec : list of jax.numpy.ndarray
        Wavevectors.

    """
    freq_period = 1
    if spacing is not None:
        freq_period = 2 * np.pi / spacing

    kvec = []
    for axis, s in enumerate(shape[:-1]):
        k = np.fft.fftfreq(s).astype(dtype) * freq_period
        kvec.append(k)

    k = np.fft.rfftfreq(shape[-1]).astype(dtype) * freq_period
    kvec.append(k)

    kvec = np.meshgrid(*kvec, indexing='ij', sparse=True)

    return kvec

def cic_preprocess(skewers_fin,nc):
    part = skewers_fin.reshape(1,-1,3)
    shape = [nc,nc,nc]
    nx, ny, nz = shape[0], shape[1], shape[2]
    ncc = [nx, ny, nz]


    if len(part.shape) > 3:
        part = np.reshape(part, (batch_size, -1, 3))

    part = np.expand_dims(part,2)
    floor = np.floor(part)
    connection = np.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1], [1., 1, 0],
                      [1., 0, 1], [0., 1, 1], [1., 1, 1]]])


    neighboor_coords =floor + connection
    kernel = 1. - np.abs(part - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords=np.array(neighboor_coords,dtype=np.int32)
    neighboor_coords = np.mod(neighboor_coords, ncc)

    na = neighboor_coords[0,:,:].reshape(-1,3)
    naa = ny**2*na[:,0]+nx*na[:,1]+na[:,2]
    return naa,kernel


#various MUSE-related functions


def _split_rng(rng, N):
    keyz = []
    for i in range(N):
        rng, subkey = jax.random.split(rng)
        keyz.append(rng)
    return keyz

def ravel_θ(θ):
    _ravel_θ, _unravel_θ = _ravel_unravel(θ)
    return _ravel_θ(θ)

def unravel_θ(θ):
    _ravel_θ, _unravel_θ = _ravel_unravel(θ)

    return _unravel_θ(θ)

def ravel_z(z):
    _ravel_z, _unravel_z = _ravel_unravel(z)
    return _ravel_z(z)

def unravel_z(z):
    _ravel_z, _unravel_z = _ravel_unravel(z)

    return _unravel_z(z)


def _ravel_unravel(x):
    if isinstance(x, (tuple,list)):
        i = 0
        slices_shapes = []
        for elem in x:
            if isinstance(elem, Number):
                slices_shapes.append((i, None))
                i += 1
            else:
                slices_shapes.append((slice(i,i+elem.size), elem.shape))
                i += elem.size
        ravel = lambda tup: np.concatenate(tup, axis=None)
        unravel = lambda vec: tuple(vec[sl] if shape is None else vec[sl].reshape(shape) for (sl,shape) in slices_shapes)
    elif isinstance(x, Number):
        ravel = lambda val: np.array([val])
        unravel = lambda vec: vec.item()
    elif isinstance(x, dict):
        keys = x.keys()
        ravel_to_tup = lambda dct: tuple(dct[k] for k in keys) if isinstance(dct, dict) else dct
        unravel_tup = lambda tup: {k: v for (k, v) in zip(keys, tup)}
        ravel_tup, unravel_to_tup = _ravel_unravel(ravel_to_tup(x))
        ravel = lambda dct: ravel_tup(ravel_to_tup(dct))
        unravel = lambda vec: unravel_tup(unravel_to_tup(vec))
    else:
        ravel = unravel = lambda z: z
    return (ravel, unravel)

