# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:09:53 2021

@author: optimus
"""
import numpy as np

def convolutionalEncoder(bitsIn, FSM):
    assert len(bitsIn) % FSM.numberOfBitsIn == 0, "Input to convolutional encoder must be an integer multiple of FSM.numberOfBitsIn"
    numberOfSteps = bitsIn // FSM.numberOfBitsIn
    i = 0
    # Empty array
    encodedStream = np.array()
    while i < numberOfSteps:
        output = FSM.step(bitsIn[i * FSM.numberOfBitsIn, (i + 1) * FSM.numberOfBitsIn])
        encodedStream = np.hstack((encodedStream, output))
    
    return encodedStream