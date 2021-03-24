# turboDNA

Getting started:
It is optional (but reccomended) that you use the spec-file.txt to generate a conda environment to run the modules in this repo.

convolutional.py is intended as a module for encoding using a finite state machine and decoding using one of: Viterbi, forward algorithm, backwards algorithm.
See <insert citation here> for some good text on hidden Markov model decoding.

polar.py is intended as a module for encoding using polar code construction as explained in release 15 of 5G new radio <insert citation here>
Decoding could be using sequencial cancelation, or using the factor graph of the construction and then with the min-sum decoder.

