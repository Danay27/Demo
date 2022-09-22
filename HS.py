import copy
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from HS.HSIndividual import HSIndividual


class HarmonySearch:
    '''
    The class for harmony search algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params, function):
        '''
        :param sizepop: population sizepop
        :param vardim: dimension of variables
        :param bound: boundaries of variables
        :param MAXGEN: termination condition
        :param params: algorithm required parameters, it is a list which is consisting of [HMCR,PAR,bw]
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.function = function
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))    # HMS个一维数组
        self.trace = np.zeros((self.MAXGEN, 3))       # 每次迭代产生的三个数据组成一个3X1的数组

    def initialize(self):
        '''
        initialize the population of hs
        :return:
        '''
        for i in range(0, self.sizepop):
            ind = HSIndividual(self.vardim, self.bound, self.function)
            ind.generate()
            self.population.append(ind)

    def evaluation(self):
        '''
        evaluation the fitness of the population
        :return:
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def improvise(self):
        '''
        improvise a new harmony
        :return:
        '''
        ind = HSIndividual(self.vardim, self.bound, self.function)

        for i in range(0, self.vardim):
            r1 = random.random()
            r2 = random.random()
            if r1 <= self.params[0]:
                if r2 <= self.params[1]:
                    if r2 > 0.5:
                        ind.chrom[i] = self.population[random.randint(0, self.sizepop - 1)].chrom[i] + r2 * self.params[2]
                    else:
                        ind.chrom[i] = self.population[random.randint(0, self.sizepop - 1)].chrom[i] - r2 * self.params[2]
                else:
                    ind.chrom[i] = self.population[random.randint(0, self.sizepop - 1)].chrom[i]
            else:
                ind.chrom[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * random.random()

        ind.calculateFitness()
        return ind

    def update(self, ind):
        '''
        update harmony memory
        :param ind:
        :return:
        '''
        minIdx = np.argmin(self.fitness)
        if ind.fitness < self.population[minIdx].fitness:
            self.population[minIdx] = ind
            self.fitness[minIdx] = ind.fitness

        # maxIdx = np.argmax(self.fitness)
        # if ind.fitness > self.population[maxIdx].fitness:
        #     self.population[maxIdx] = ind
        #     self.fitness[maxIdx] = ind.fitness
            # z = self.population[maxIdx].fitness
            # print("=====更新和声库=====")
            # print("在位置：" + str(maxIdx) + "上更改")
            # print("解向量由：" + str(self.population[maxIdx].chrom) + "更改为：" + str(ind.chrom))
            # print("适应度函数由：" + str(self.population[maxIdx].fitness) + "更改为：" + str(ind.fitness))
            # print(z, "update", "to", ind.fitness)

    def solve(self):
        '''
        the evolution of the hs algorithm
        :return:
        '''
        # print("开始执行HS")
        start = time.time()
        self.t = 0
        self.initialize()
        self.evaluation()
        # print("===初始解向量===")
        # for i in range(self.sizepop):
        #     print(self.population[i].chrom)
        # print("=========")
        # best = np.max(self.fitness)
        # bestIndex = np.argmax(self.fitness)

        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)

        worstIndex = np.argmax(self.fitness)
        self.worst = copy.deepcopy(self.population[worstIndex])



        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        self.trace[self.t, 2] = self.worst.fitness
        # print("Generation %d: optimal function value is: %f; average function is: %f" % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        # print("=====当前最小值======")
        # print(self.trace[self.t, 0])
        # print("=====当前最大值======")
        # print(self.trace[self.t, 2])
        # print("===================")

        while self.t < self.MAXGEN - 1:
            if (self.best.fitness != 0.000):
                self.t += 1
                ind1 = self.improvise()
                self.update(ind1)

                # best = np.max(self.fitness)
                # bestIndex = np.argmin(self.fitness)
                # if best > self.best.fitness:
                #    self.best = copy.deepcopy(self.population[bestIndex])

                # best = np.min(self.fitness)
                # bestIndex = np.argmin(self.fitness)
                # if best < self.best.fitness:
                #    self.best = copy.deepcopy(self.population[bestIndex])
                # self.avefitness = np.mean(self.fitness)

                # self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
                # self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness

                bestIndex = np.argmin(self.fitness)
                self.best = self.population[bestIndex]
                self.avefitness = np.mean(self.fitness)

                worstIndex = np.argmax(self.fitness)
                worst = copy.deepcopy(self.population[worstIndex])

                self.trace[self.t, 0] = self.best.fitness
                self.trace[self.t, 1] = self.avefitness
                self.trace[self.t, 2] = worst.fitness
            else:
                print("rate:", self.t)
                break
        # print("Optimal function value is: %f; " % self.trace[self.t, 0])
        # print("Optimal solution is:")
        # for i in range(self.sizepop):
        #    print(self.population[i].chrom, self.population[i].fitness)
        # print(self.best.chrom)
        end = time.time()
        # print("===========")
        # print("time:", end - start)
        # print("HS-best:", self.best.fitness)
        # print("==============")
        # print(self.trace[self.t, 0])
    #     self.printResult()
    # def printResult(self):
    #     '''
    #     plot the result of abs algorithm
    #     :return:
    #     '''
    #     x = np.arange(0, self.MAXGEN)
    #     y1 = self.trace[:, 0]
    #     y2 = self.trace[:, 1]
    #     plt.plot(x, y1, 'r', label='optimal value')
    #     plt.plot(x, y2, 'g', label='average value')
    #     plt.xlabel("Iteration")
    #     plt.ylabel("function value")
    #     plt.title("Harmony search algorithm for function optimization")
    #     plt.legend()
    #     plt.show()

