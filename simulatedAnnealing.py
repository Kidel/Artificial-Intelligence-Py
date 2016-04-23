from math import log, exp
from random import random

class SimulatedAnnealing(object):
    def __init__(self, problem):
        #defining parameters
        self.infinite = 100000.0
        self.startingTemp = 100.0
        self.problem = problem
        
    def cooling(self, time): 
        # linear: self.startingTemp * (((-1 / self.infinite) * (time + 1)) + 1)
        # logarithmic: self.startingTemp * (1 / (1 + log(1+time)))
        return self.startingTemp * ((((-1 / self.infinite) * (time + 1)) + 1) * (1 / (1 + log(1+time))))
    
    def callback(self, currentNode, evaluation, temp):
        self.evaluations.append(evaluation)
        self.nodes.append(currentNode)
        self.temps.append(temp)
        
    def solve(self):
        self.evaluations = []
        self.nodes = []
        self.temps = []
        self.iterations = 0
        currentNode = self.problem.createRandomNode()
        
        for time in range(0, int(self.infinite)):
            evaluation = self.problem.evaluate(currentNode)
            
            temp = self.cooling(time)
            if temp == 0:
                print("temperature reached 0")
                self.localMax = currentNode
                return self.localMax
            
            self.callback(currentNode, evaluation, temp)
            
            if self.problem.halting(evaluation):
                print("halted")
                self.localMax = currentNode
                return self.localMax
            
            nextNode = self.problem.selectRandomSuccessor(currentNode)
            deltaE = self.problem.evaluate(nextNode) - evaluation
            if deltaE >= 0:
                currentNode = nextNode
            else: 
                prob = exp(deltaE/temp)
                rand = random()
                if rand < prob:
                    currentNode = nextNode
            self.iterations += 1
        
        self.localMax = currentNode
        return self.localMax