from random import randint, shuffle
import matplotlib.pyplot as plt
import numpy as np

class QueensPuzzle(object):
    def __init__(self, size=8):
        self.name = "Queens Puzzle"
        self.size = size
    
    def getInitialNode(self):
        board = [None] * self.size
        for i in range(0, self.size):
            board[i] = i
        return board
    
    def randomSwap(self, board):
        newBoard = list(board)
        c1 = randint(0, self.size - 1)
        c2 = randint(0, self.size - 1)
        app = newBoard[c1]
        newBoard[c1] = newBoard[c2]
        newBoard[c2] = app
        return newBoard
    
    def createRandomNode(self):
        board = self.getInitialNode()
        shuffle(board)
        return board
    
    def selectRandomSuccessor(self, board):
        return self.randomSwap(board)
    
    def evaluate(self, board):
        conflicts = 0;
        for c in range(0, self.size):
            for j in range(0, self.size):
                if c!=j and board[c]==board[j]:
                    conflicts += 1
                if c!=j and (abs(c-j)==abs(board[c]-board[j])):
                    conflicts += 1
        return (-1)*conflicts
    
    def halting(self, evaluation):
        return (evaluation == 0)
    
    def fitness(self, node):
        return self.evaluate(node)
    
    def mutation(self, node):
        i = randint(0, len(node)-1)
        node[i]=randint(0, len(node)-1)
    def printBoard(self, board):
        size = len(board)
        image = np.zeros(size**2)
        # alternate pattern
        for i in range(0, size**2, 2):
            if((i//size) % 2):
                image[i] = 10
            else:
                image[i+1] = 10
        # populating with queens
        for i in range(0, size):
            image[i*(size) + board[i]] = 150
        imageT = np.zeros(size**2)
        # translating
        for i in range(0, size**2):
            imageT[((i%size)*size) + (i//size)]=image[i]
        image = imageT
        # reshape things
        image = image.reshape((size, size))
        row_labels = range(size)
        col_labels = range(size)
        plt.matshow(image)
        plt.xticks(range(size), col_labels)
        plt.yticks(range(size), row_labels)
        plt.show()