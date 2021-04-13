# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:09:53 2021

@author: Omer Sella
"""
import numpy as np
import copy



class FSM():
    def __init__(self, states, triggers, outputTable, transitionTable, initialState):
        
        
        # Safety: all triggers must be of the same fixed length
        fixedLength = len(triggers[0])
        assert all(len(t) == fixedLength for t in triggers)
        
        self.numberOfStates = len(states)
        self.stepSize = fixedLength
        self.transitionTable = transitionTable
        self.outputTable = outputTable
        self.triggerDictionary = self.generateDictionary(triggers)
        self.stateDictionary = self.generateDictionary(states)
        self.presentState = initialState
        self.presentStateCoordinate = initialState
        
    def checkStreamLength(self, length):
        result = False
        if length % self.stepSize == 0:
            result = 'OK'
        return result
    
    def generateDictionary(self, keys):
        newDictionary = dict()
        #entries = np.arange(0, self.outputTable.shape[0])
        i = 0
        for t in keys:
            #assert t not in newDictionary,  'Multiple appearences for trigger'
            newDictionary[str(t)] = i
            i = i + 1
        return newDictionary
    
    def step(self, trigger):
        output = 'Fail'
        triggerCoordinate = self.triggerDictionary[str(trigger)]
        output = self.outputTable[self.presentStateCoordinate][triggerCoordinate]
        nextState = self.transitionTable[self.presentStateCoordinate, triggerCoordinate]
        nextStateCoordinate = self.stateDictionary[str(nextState)]
        self.presentState = nextState
        self.presentStateCoordinate = nextStateCoordinate
        #outputSerialized = 
        return output
    
    def getNextPossibleStates(self, state):
        nextStates = self.transitionTable[state, :]
        return nextStates
    
    def getNextPossibleOutputs(self, state):
        nextOutputs = self.outputTable[state]
        return nextOutputs
    
def trellisGraphics(numberOfStates):
    # Omer Sella: still under construction
    states = np.arange(numberOfStates)                
    timeStamp = np.ones(self.numberOfStates)
    colours = np.arange(self.numberOfStates)
    sizeOfState = 4
    fig, ax = plt.subplots()
    scatter = ax.scatter(states, timeStampe, c=colours, s=sizeOfState)
    plt.show()    
    
    
def convolutionalEncoder(streamIn, FSM, graphics = False):
    
    assert FSM.checkStreamLength(len(streamIn)) == 'OK' , "Input to convolutional encoder must be an integer multiple of FSM.numberOfBitsIn"
    numberOfSteps = len(streamIn) // FSM.stepSize
    i = 0
    encodedStream = []
    while i < numberOfSteps:
        trigger = streamIn[i * FSM.stepSize: (i + 1) * FSM.stepSize]
        #print(trigger)
        trigger = list(trigger)
        print(trigger)
        output = FSM.step(trigger)
        print(output)
        encodedStream.append(output)
        i = i + 1
        #print(encodedStream)
    return encodedStream


class path():
    def __init__(self, initialState):
        self.traversedStates = [initialState]
        self.scores = [0]
        self.presentScore = 0
        self.pathTriggers = []
        
    def presentState(self):
        return self.traversedStates[-1]
    
    def presentScore(self):
        return self.currentScore
    
    def step(self, extension):
        nextState = extension[0]
        trigger = extension[1]
        addedScore = extension[2]
        print("*** before step:")
        print("*** " + str(self.pathTriggers))
        print("*** " + str(self.traversedStates))
        print("*** ")
        print("*** ")
        self.pathTriggers.append(trigger)
        self.scores.append(addedScore)
        self.traversedStates.append(nextState)   
        self.presentScore = self.presentScore + addedScore
        return
        


def viterbiDecoder(numberOfStates, initialState, scoreFunction, observedSequence, symbolsPerStateTransition):
    # Viterbi decoder inspired by the implementation suggested by Todd K. Moon, programming laboratory 10, page 528.
    # More explanations on the Viterbi decoder are found on page 473 of the same book.
    # A metric function (!) that accepts a set of states p, next state q and observed stream r,
    # and returns the branch metric present state, next state and returns 
    assert len(observedSequence) % symbolsPerStateTransition == 0
    newPath = path(initialState)
    paths = [newPath]
    i = 0
    while i < len(observedSequence) // symbolsPerStateTransition:
        print("*** i is :")
        print(i)
        observedOutput = observedSequence[i * symbolsPerStateTransition : (i + 1) * symbolsPerStateTransition]
        print(observedOutput)
        newPaths = []
        
        for p in paths:
            extensions = scoreFunction(p.presentState(), observedOutput, i)
            
            for extension in extensions:
                #print(extension)    
                newPath = path(0)
                newPath = copy.deepcopy(p)
                newPath.step(extension)
                #print(newPath.traveresedStates)
                newPaths.append(newPath)
        paths = newPaths
        i = i + 1
    return paths
    
def genericScoreFunction(myFSM, presentState, observedOutput, timeStep, additionalInformation):
    # some useful distances in this package, we may want to try and use it.
    from scipy.spatial import distance
    nextPossibleStates = myFSM.getNextPossibleStates(presentState)
    nextPossibleOutputs = myFSM.getNextPossibleOutputs(presentState)
    extensions = []    
    nextPossibleScores = []
    for output in nextPossibleOutputs:
        # compute the score of output with respect to the observedOutput
        #print("*** output is:")
        #print(output)
        #print("*** observedOutput is:")
        #print(observedOutput)
        score = 1 - distance.hamming(output, observedOutput)
        nextPossibleScores.append(score)
    extensions = []
    # Omer Sella: safety
    assert (len(nextPossibleStates) == len(nextPossibleOutputs))
    for i in range(len(nextPossibleStates)):
        extensions.append( [nextPossibleStates[i], nextPossibleOutputs[i], nextPossibleScores[i]])
    #print("*** extensions are: ")
    #print(extensions)
    return extensions


    
def exampleTwoThirdsConvolutional():
    states = [0,1,2,3,4,5,6,7]
    triggers = [[0,0], [0,1], [1,0], [1,1]]
    nextStateTable = np.array([[0,1,2,3], [4,5,6,7], [1,0,3,2], [5,4,7,6], [2,3,0,1], [6,7,4,5] , [3,2,1,0], [7,6,5,4] ])
    outputTable = [[[0,0,0], [1,0,0], [0,1,0], [1,1,0]], 
                   [[0,0,1], [1,0,1], [0,1,1], [1,1,1]],
                   [[1,0,0], [0,0,0], [1,1,0], [0,1,0]],
                   [[1,0,1], [0,0,1], [1,1,1], [0,1,1]],
                   [[0,1,0], [1,1,0], [0,0,0], [1,0,0]],
                   [[0,1,1], [1,1,1], [0,0,1], [1,0,1]],
                   [[1,1,0], [0,1,0], [1,0,0], [0,0,0]],
                   [[1,1,1], [0,1,1], [1,0,1], [0,0,1]]]
    initialState = 0
    myFSM = FSM(states, triggers, outputTable, nextStateTable, initialState)
    stream = np.random.randint(0,2,10)
    encodedStream = convolutionalEncoder(stream, myFSM)

    flatStream = []
    for sublist in encodedStream:
        for item in sublist:
            flatStream.append(item)

    def myScoreFunction(state, observation, time):
        return genericScoreFunction(myFSM, state, observation, time, None)

    paths = viterbiDecoder(8, 0, myScoreFunction, flatStream, 3)

    return encodedStream, paths

def testConvolutional_2_3():
    status = 'Not working'
    stream, encodedStream = exampleTwoThirdsConvolutional()
    if len(encodedStream) == len(stream)//2:
        status = 'OK'
    return status

es, paths = exampleTwoThirdsConvolutional()
