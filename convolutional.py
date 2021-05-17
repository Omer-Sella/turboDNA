# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:09:53 2021

@author: Omer Sella
"""
import numpy as np
import copy
BIG_NUMBER = np.Inf


class FSM():
    
    """
    A finite state machine (FSM) a graph made of states (vertices) and a connectivity matrix (transitionTable).
    If the FSM is at state s0, then on input (trigger) i, it will emmit an output e, and transition from state 
    s0 (the present state) to a next state, determined by the transitionTable.
    An initialState has to be given so that we know how to start.
    
    """
    def __init__(self, states, triggers, outputTable, transitionTable, initialState):
        # Safety: all triggers must be of the same fixed length
        fixedLength = len(triggers[0])
        assert all(len(t) == fixedLength for t in triggers)
    
        self.triggers = triggers
        self.numberOfStates = len(states)
        self.stepSize = fixedLength
        self.transitionTable = transitionTable
        self.outputTable = outputTable
        # Enumerate the possible triggers according to the order they were given.
        self.triggerDictionary = self.generateDictionary(triggers)
        
        # Enumerate the possible states according to the order they were given.
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
        #Init output
        output = ''
        #Get the number of the trigger
        triggerCoordinate = self.triggerDictionary[str(trigger)]
        #Get the output from the outputTable, matching to the number of state we are in and the number of the trigger.
        output = self.outputTable[self.presentStateCoordinate][triggerCoordinate]
        nextState = self.transitionTable[self.presentStateCoordinate, triggerCoordinate]
        nextStateCoordinate = self.stateDictionary[str(nextState)]
        self.presentState = nextState
        self.presentStateCoordinate = nextStateCoordinate
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
    
    
def FSMEncoder(streamIn, FSM, graphics = False):
    """
    streamIn is a stream of (symbols). There is no safety over their content validity,
    but we do verify that the stream of symbols could be chopped into an integer number of triggers.
    For example: if the stream is made of bits, and we need 5 bits per trigger, then the stream has to have length = 5*k for some k.
    """
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
        self.pathEmitted = []
        
    def presentState(self):
        return self.traversedStates[-1]
    
    
    def appendToPath(self, extension):
        nextState = extension[0]
        trigger = extension[1]
        nextOutput = extension[2]
        addedScore = extension[3]
        print("*** before step:")
        print("*** " + str(self.pathTriggers))
        print("*** " + str(self.traversedStates))
        print("*** ")
        print("*** ")
        self.pathTriggers.append(trigger)
        self.pathEmitted.append(nextOutput)
        self.scores.append(addedScore)
        self.traversedStates.append(nextState)   
        self.presentScore = self.presentScore + addedScore
        return
        


def viterbiDecoder(numberOfStates, initialState, fanOutFunction, observedSequence, symbolsPerStateTransition):
    # Viterbi decoder inspired by the implementation suggested by Todd K. Moon, programming laboratory 10, page 528.
    # More explanations on the Viterbi decoder are found on page 473 of the same book.
    # A metric function (!) that accepts a set of states p, next state q and observed stream r,
    # and returns the branch metric present state, next state and returns 
    
    # There is no safety of the content of the observedSequence, but the observedSequence has to be chopped into observable state transitions.
    # This means that *this version of the Viterbi decoder does not support insertions or deletions.
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
            # Omer Sella: fix the line below - a scorFunction is expected to give a number, not a triplet.
            
            extensions = fanOutFunction(p.presentState(), observedOutput, i)
            
            for extension in extensions:
                #print(extension)    
                newPath = path(0)
                newPath = copy.deepcopy(p)
                newPath.appendToPath(extension)
                #print(newPath.traveresedStates)
                newPaths.append(newPath)
        paths = newPaths
        #Omer Sella: Here there is usually pruning, i.e.: getting rid of costly candidates, but not in this version.
        i = i + 1
    
    #Omer Sella: Now let's find "the" most likely path, which is the that has the LOWEST score (so score is like loss)
    lowestScore  = BIG_NUMBER
    numberOfEquallyMostLikely = 1
    for p in paths:
        if p.presentScore < lowestScore:
            lowestScore = p.presentScore
            mostLikelyPath = p
        else:
            if p.presentScore == lowestScore:
                numberOfEquallyMostLikely = numberOfEquallyMostLikely + 1

    # Omer Sella: Viterbi is supposed to return the original input, it could also return paths
    # So we first return the most likely path, if there is more than one then numberOfEquallyMostLikely will be > 1
    # Then we return all paths 
    return mostLikelyPath, numberOfEquallyMostLikely, paths
    
def genericFanOutFunction(myFSM, presentState, observedOutput, timeStep, additionalInformation):
    # some useful distances in this package, we may want to try and use it.
    from scipy.spatial import distance
    nextPossibleStates = myFSM.getNextPossibleStates(presentState)
    nextPossibleOutputs = myFSM.getNextPossibleOutputs(presentState)
    triggers = myFSM.triggers
    extensions = []    
    nextPossibleScores = []
    for output in nextPossibleOutputs:
        # compute the score of output with respect to the observedOutput
        # print("*** output is:")
        # print(output)
        # print("*** observedOutput is:")
        # print(observedOutput)
        # Omer Sella: scipy.distance.hamming(x,y) return number_of_coordinates_where_different / number_of_coordinates. Sequences must have identical length.
        score = distance.hamming(output, observedOutput)
        print(score)
        nextPossibleScores.append(score)
    extensions = []
    # Omer Sella: safety
    assert (len(nextPossibleStates) == len(nextPossibleOutputs))
    for i in range(len(nextPossibleStates)):
        extensions.append( [nextPossibleStates[i], triggers[i], nextPossibleOutputs[i], nextPossibleScores[i]])
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
    encodedStream = FSMEncoder(stream, myFSM)

    flatStream = []
    for sublist in encodedStream:
        for item in sublist:
            flatStream.append(item)

    def myFanOutFunction(state, observation, time):
        return genericFanOutFunction(myFSM, state, observation, time, None)

    mostLikelyPath, numberOfEquallyLikelyPaths, paths = viterbiDecoder(8, 0, myFanOutFunction, flatStream, 3)


    return stream, encodedStream, paths, mostLikelyPath, numberOfEquallyLikelyPaths

def testConvolutional_2_3():
    status = 'Not working'
    stream, encodedStream = exampleTwoThirdsConvolutional()
    if len(encodedStream) == len(stream)//2:
        status = 'OK'
    return status

stream, encodedStream, paths, mostLikelyPath, numberOfEquallyLikelyPaths = exampleTwoThirdsConvolutional()

if __name__ == '__main__':
    stream, encodedStream, paths, mostLikelyPath, numberOfEquallyLikelyPaths = exampleTwoThirdsConvolutional()