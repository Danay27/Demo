import random
import numpy as np
import ObjFunction

'''
generate a solution vector
'''


class HSIndividual:
    '''
    individual of harmony search algorithm
    '''

    def __init__(self, vardim, bound, function="Step"):
        '''
        :param vardim: dimension of variables
        :param bound: boundaries of variables
        :param n:
        '''
        self.vardim = vardim
        self.bound = bound
        self.function = function
        self.chrom = np.zeros(self.vardim)       #大小为vardim的浮点数数组，以数字0填充,例[0. 0. 0. 0. 0.]
        self.fitness = 0.

    def generate(self):
        '''
        generate a random chromosome for harmony search algorithm
        :return:
        '''
        # rnd = np.random.random(size = self.vardim)
        for i in range(0, self.vardim):
            self.chrom[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * random.random()

    def calculateFitness(self):
        '''
        calculate the fitness of the chromosome
        :return:
        '''
        f = getattr(ObjFunction, self.function)
        self.fitness = f(self.vardim, self.chrom, self.bound)
