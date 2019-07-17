import numpy as np
from gym.spaces.box import Box
import itertools

class Discretizer():
    def __init__(self):
        pass

    def discretize(self, space):
        raise NotImplementedError

class Box_Discretizer(Discretizer):
    def __init__(self, box_space, N=10):
        '''Creates a conversion class for continuous Box space to convert to discrete space.
        Takes: a box_space and the number of buckets (either an integer or list of integers equal to the shape of box_space)
        '''
        assert isinstance(box_space, Box)
        self.shape = box_space.shape
        self.box = box_space
        self.lbound = box_space.low
        self.ubound = box_space.high

        if isinstance(N, int):
            N = np.ones(self.shape).astype(int) * N
        else:
            if isinstance(N, list):
                N = np.array(N)
            assert N.shape == box_space.shape

        max_buckets = self.ubound - (self.ubound - self.lbound) / N

        step_size = (self.ubound - self.lbound) / N

        self.buckets = [np.arange(self.lbound[i], self.ubound[i], step_size[i]) for i in range(N.size)]

    def discretize(self, space):
        ''' Converts an input space into the index of the buckets for each dimension of the Box.
        Takes: an array of the state space.
        Returns: an array of which bucket each array dimension fall in.
        '''
        if type(space) == tuple or type(space) == list:
            space = np.array(space)

        assert space.shape == self.box.shape

        discrete_space = np.array([self.buckets[i][np.digitize(s, self.buckets[i], right=True)-1]
                                   for i, s in enumerate(space)])
        return discrete_space

    def list_all_states(self):
        ''' Returns all possible state spaces defined by the box bucekts.'''
        all_states = [l for l in itertools.product(*[b for b in self.buckets])]
        return all_states