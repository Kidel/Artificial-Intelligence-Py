from random import randint, shuffle

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