"""Instance-uniform foreground sampling with an inverse-distance spatial prior."""
import numpy as np


def inverse_distance_probabilities(coordinates,center):
    xy=np.asarray(coordinates,dtype=np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or not len(xy):raise ValueError('Expected nonempty foreground y,x coordinates')
    w=1/(1+np.linalg.norm(xy-np.asarray(center,dtype=np.float64),axis=1))
    return w/w.sum()


def draw_schedule(groups,seed=111,epochs=120,draws=128):
    """Choose an instance uniformly, then a fresh foreground point at every draw."""
    rng=np.random.default_rng(seed);result=np.empty((epochs,draws),np.int64)
    for epoch in range(epochs):
        for draw in range(draws):
            group=groups[int(rng.integers(len(groups)))]
            index=int(rng.choice(len(group['indices']),p=group['probabilities']))
            result[epoch,draw]=group['indices'][index]
    return result
