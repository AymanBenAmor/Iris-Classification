import numpy as np
x_entrer = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float)
x_entrer = x_entrer/np.amax(x_entrer , axis=0)
x = np.split(x_entrer,[8])[0]
xPrediction = np.split(x_entrer,[8])[1]

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.w1 = np.random.randn(self.inputSize , self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self,x):
        self.z = np.dot(x,self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.w2)
        o = self.sigmoid(self.z3)
        return o
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    def sigmoidPrime(self,s):
        return s*(1-s)
    def backward(self,x,y,o):
        self.o_error = y-o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self,x,y):
        o = self.forward(x)
        self.backward(x,y,o)
    def predict(self):
        print("donnee predite apres entrainn")
        print("entree: \n" + str(xPrediction))
        print("entree: \n" + str(self.forward(xPrediction)))

        if(self.forward(xPrediction) < 0.5):
            print("la fleur est bleu \n")
        else:
            print("la fleur est rouge \n")



MN = Neural_Network()

for i in range(30000):
    print("#"+str(i)+"\n")
    print("valeurs d'entrees: \n"+str(x))
    print("sorti actuelle: \n" + str(y))
    print("sorti predite: \n" + str(np.matrix.round(MN.forward(x),2)))
    print("\n")
    MN.train(x,y)

MN.predict()
