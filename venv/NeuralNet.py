import numpy as np
import random
import math
from LanguageSys import*

class NeuralNet:
    def __init__(self,x,y,inputs):
        self.x = x
        self.y = y
        self.inputs = inputs

        self.neurons = self.generateMatrix(x,y)
        self.weights = self.generateWeights(x,y, inputs)

        self.lastinput = [] #safes last input in order to match with winning neuron i.e. for punishment or enforcement
        self.winning = []  #x-y Position of winning neuron
        self.distances = [] #matrix of distances as int values
        self.gaussian_dist = [] #matrix of learning affection on the area with gaussian function

        self.habitus = [random.uniform(0, 1) for k in range(3)]
        self.width = 2
        self.fitness = 0  #here we can punish or encourage behaviour
        self.positions = self.posfunc()# gives us all positions of our neurons as matrix of tupels (NEVER CHANGED)


    def generateMatrix(self, x, y):  # Generates random matrix, x =  Number of Neurons /  and y = Number of Inputs
        return np.random.uniform(0,1, (y,x))

    def generateWeights(self, x, y, inputs):
        return np.random.uniform(0,1, (y,x,inputs))

    def generateVector(self, x):  # Generates random vector, x =  number of entries
        return np.random.uniform(0,1, x)

    ### NOW WE DO THE CALCULATIONS

    def updateNeurons(self, inputs):  #range = neighbourhood-steps /// range = 0 means no learning except the selected
        self.lastinput = inputs
        self.compareWeights(inputs)   #updates neuron-Values
        self.distances = self.distance_from_win()
        self.gaussian_dist = self.gaussian()

        #self.neighbours = self.neighboursfunc()
        self.learning(inputs)
        if self.width > 1.1:
            self.width -= 0.1

    def compareWeights(self, input): #compares and returns matrix of similarity values
        # update all the neurons:
        self.neurons = np.sum(input - self.weights, axis=2)

        # find the indices (as a tuple) of the neuron with the lowest absolute excitation
        self.winning = np.array(np.unravel_index(np.argmin(np.abs(self.neurons)), self.neurons.shape))
        print (self.winning, "WINNING")

    def posfunc(self):

        arr = np.array([[i, j] for i in range(self.y) for j in range(self.x)])

        return arr

    def distance_from_win(self): #returns distances from current winning neuron

        distance_vectors = self.positions - self.winning
        scalar_distances = np.power(np.sqrt(distance_vectors * distance_vectors),2)
        scalar_distances = np.sum(scalar_distances, axis =1)
        scalar_distances = np.round(np.sqrt(scalar_distances))

        return scalar_distances

    def gaussian(self): #GETS ARRAY OF INT-VALUES AND RETURNS GAUSSIAN VALUES WITH CURRENT WIDTH
        pos = self.distances
        pos = np.power(pos, 2)
        pos = pos * (-1/self.width)
        e = np.around(np.power(math.e, pos), decimals=3)
        gaussian = (1/ math.sqrt(2*math.pi))* e
        return gaussian

    def learning (self, inputs):
        difference = inputs-self.weights #negative == weight is too big
        #gauss_diff = np.reshape(self.gaussian_dist, self.x*self.y, self.inputs)
        difference = np.reshape(difference, ( self.inputs, self.x*self.y)) #TODO: MAKE WEIGHTS IN THIS FORMAT FROM BEGINNING
        influence = difference *self.gaussian_dist
        influence = np.reshape(influence,(self.y, self.x,self.inputs)) #TODO: CHECK THIS
        self.weights += influence



    def speak(self):
        myLanguage = LanguageSys()
        return myLanguage.output(self.habitus, self.winning)

    def shapeInput(self, myarray): #shapes input array into vector, needs at least 1 argument
        first = [myarray[0]]
        matrix = [first]
        for i in range (len(myarray)):
            if i != 0:
                next = [myarray[i]]

                matrix = np.append(matrix, [next], axis=0)
        return matrix

    def posval(self, pos): #for testing
        if pos[0] == 0:
            if pos[1] == 0:
                val1 = 0
                val2 = 1

            else:
                val1 = 1
                val2 = 0
        else:
            if pos[1] == 0:
                val1 = 0.1
                val2 = 0.1
            else:
                val1 = 0.01
                val2 = 0.001
        return [val1, val2]




    def punish (self, punishment): #punishes winning weights
        for i in range (self.inputs):

            self.weights[self.winning[0]][self.winning[1]][i][0] -= punishment

    def punishwin(self, distance): #in case we want to decrement weights of winning neuron
        distance =1+ (distance/10)
        for i in range (self.inputs):
            weight = self.weights[self.winning[0]][self.winning[1]][i][0]
            input = self.lastinput[i][0]

            if weight > input:
                self.weights[self.winning[0]][self.winning[1]][i][0] += (weight - input)*distance
            else:
                self.weights[self.winning[0]][self.winning[1]][i][0] -= (input - weight)*distance

    def enforce(self): #in case got what we want
        for i in range (self.inputs):
            weight = self.weights[self.winning[0]][self.winning[1]][i][0]
            input = self.lastinput[i][0]

            if weight > input:
                self.weights[self.winning[0]][self.winning[1]][i][0] -= (weight - input)
            else:
                self.weights[self.winning[0]][self.winning[1]][i][0] += (input - weight)






if __name__ == '__main__':

    inputs = 5
    myRange = 2

    n = NeuralNet(4,4,inputs)

    inputVals = np.array([.0, .0, .0, .0, 0.]) #Generating some Input

    data = [[1.00, 0.46, 0.1864695139],[0.96, 0.36 ,0.03775425955],[0.23, 0.49, 1.284095174],[0.19, 0.56, 0.07685956616],[0.15,1.34,0.1998048488],[0.14,0.59,	0.03390127349],[0.12,0.40,	0.03007330681],[0.10,0.53,	0.2196952638],[0.09,0.38,	0.2043333584],[0.09,0.29,	0.8573394381],[0.07,0.57,	0.08091270735],[0.07,0.57,	0.01491155646],[0.07,1.16,	0.07708474067],[0.07,0.22,	0.04656108484],[0.06,0.38,	0.9931697065]]
    countries = ["China","Indien", "Vereinigte Staaten von Amerika","Indonesien" ,"Brasilien" ,"Pakistan",  "Nigeria", "Bangladesch" ,"Russland", "Mexiko","Japan",  "Philippinen","Aethiopien",  "Aegypten",   "Vietnam", "Deutschland" ]


    for j in range (1000):
        n.updateNeurons(inputVals)

    # print (n.winning)
    print(np.round(n.weights))


    
    

