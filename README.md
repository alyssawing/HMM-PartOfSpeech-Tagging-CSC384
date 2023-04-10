# HMM-PartOfSpeech-Tagging-CSC384

This project for CSC384, Intro to AI, is an implementation of a Hidden Markov Model that can accomplish Part-Of-Speech (POS) tagging. Every word and 
punctuation can be tagged based on its syntactic role in a sentence (for example, a noun, verb, or adjective). There are 91 possible tags, including ambiguity tags. Ambiguity tags are due to words having multiple meanings depending on the context. In this implementation, the order of the tags matter, i.e. AJ0-AV0 is not recognized as the same as AV0-AJ0 in the model, but they have the same meaning.

One or more training files are used to train the model, and they are formatted with every line providing the word and its tag. For example: 

    They : PNP
    played : VVD-VVN
    volleyball : NN1
    yesterday : AV0
    . : PUN 

The equivalent test file for that sentence would then be: 

    They
    played 
    volleyball
    yesterday 
    . 

A complete list of the tags and what they represent can be found in the POS-Tags file.

## How to run the code

Ensure that all files are in the relevant folder. Here, the files are:
* training1.txt and training2.txt: training files 
* test1.txt and test2.txt: test files corresponding to the training files (i.e., test1 contains all the words without the tags from training1)
* output.txt: the output file that the model will write its answers into

The command used in terminal is: 

    python3 tagger.py --trainingfiles <training files> --testfile <test file> --outputfile <output file>
  
The code also can measure the accuracy of the model by comparing the model's predictions to the answers, if there is an answerfile provided in the main function for it to read from. The accuracies with this model found are: 
* training on set 1, testing on set 1: 97%
* training on set 2, testing on set 2: 97%
* training on set 1, testing on set 2: 85%
* training on set 2, testing on set 1: 87%
* training on sets 1&2, testing on set 1: 
* training on sets 1&2, testing on set 2: 
  
Different training and test files may also be used to test the model.

