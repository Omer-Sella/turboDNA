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
        return output
        
    
    
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
    return stream, encodedStream

def testConvolutional_2_3():
    status = 'Not working'
    stream, encodedStream = exampleTwoThirdsConvolutional()
    if len(encodedStream) == len(stream)//2:
        status = 'OK'
    return status