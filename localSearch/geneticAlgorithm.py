from math import floor
from random import randint, random

class Individual(object):
    def __init__(self, node, fitness):
        self.node = node
        self.fitness = fitness

class GeneticAlgorithm(object):
    def __init__(self, problem, populationSize=10, mutationRate=0.05, selectionRate=0.1):
        #defining parameters
        self.maxGenerations = 10000
        self.catastropheFreeGens = 0.10 * self.maxGenerations
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.selectionRate = selectionRate
        self.problem = problem
        
    def createIndividuals(self):
        self.individuals = [None] * self.populationSize
        for i in range(0, int(self.populationSize)):
            node = self.problem.createRandomNode()
            fitness = self.problem.fitness(node)
            self.individuals[i] = Individual(node, fitness)
        # order by fitness
        self.individuals.sort(key = lambda x: x.fitness, reverse=True)
        
    def mutation(self, individual):
        self.problem.mutation(individual.node)
        individual.fitness = self.problem.fitness(individual.node)
        
    def reproduction(self, individualA, individualB):
        nodeA = individualA.node
        nodeB = individualB.node
        # for each couple make 2 children with a random cut of the sequence
        while True:
            cut = randint(0, len(nodeA)-2) # -2, so at least 1 difference in the product
            if nodeA[cut] != nodeB[cut+1]: # no repetitions at least on the cut
                break
        # new nodes    
        app1 = [None] * len(nodeA)
        app2 = [None] * len(nodeB)
        for i in range(0, int(len(nodeA))):
            if i<=cut:
                app1[i]=nodeA[i]
                app2[i]=nodeB[i]
            else:
                app1[i]=nodeB[i]
                app2[i]=nodeA[i]
        
        # making new individuals from the new nodes
        individual1 = Individual(app1, self.problem.fitness(app1))
        individual2 = Individual(app2, self.problem.fitness(app2))
            
        return individual1, individual2;

    
    def callbackF(self, currentBestNode, fitness):
        self.fitnessValues.append(fitness)
        self.nodes.append(currentBestNode)
    def callbackC(self, generation):
        self.catastrophes.append(generation)
        
    def solve(self):
        self.fitnessValues = []
        self.nodes = []
        self.catastrophes = []
        self.iterations = 0
        
        # creating individuals
        self.createIndividuals()
                
        for currentGeneration in range(0, int(self.maxGenerations)):
            # adding to the fittest list for the plot
            self.callbackF(self.individuals[0].node, self.individuals[0].fitness)
            # checking if we are done prematurely
            if self.problem.halting(self.individuals[0].fitness):
                print("halted")
                self.localMax = self.individuals[0].node
                return self.individuals[0].node
            
            # catastrophe
            if ((currentGeneration > self.catastropheFreeGens) and (random() > 0.05)):
                self.catastropheFreeGens += currentGeneration + self.catastropheFreeGens;
                self.callbackC(currentGeneration)
                self.createIndividuals()
        
            # creating next generation using the fittest, discarding the others by a selection rate
            newIndividuals = []
            
            for i in range(0, len(self.individuals), 2): # by 2
                # excluding the weakest
                included = int(floor(len(self.individuals)*(1-self.selectionRate)))
                if i<included:
                    index = i
                else:
                    index = i-included
                # reproduction of 2 close individuals, generating 2 brothers
                brotherA, brotherB = self.reproduction(self.individuals[index], self.individuals[index+1])
                
                # random mutation
                if random() <= self.mutationRate:
                    self.mutation(brotherA)
                    self.mutation(brotherB)
                
                # adding the new individuals to the new generation
                newIndividuals.append(brotherA)
                newIndividuals.append(brotherB)
                
            # new generation replacing the old one
            newIndividuals.sort(key = lambda x: x.fitness, reverse=True)
            self.individuals = newIndividuals
            self.iterations += 1

        self.localMax = self.individuals[0].node
        return self.localMax