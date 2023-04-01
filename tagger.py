import os
import sys
import argparse
import numpy as np

def init_training(training_list):
    '''Given a list of training files, parse through each file to create a list 
    of lists, where each list is a sentence. Each element in each sentence is a
    mini-list where the 1st element is the word, and the 2nd element is the tag.
    
    The format of the each training file in the list is:
    word1 : tag1
    word2 : tag2 and so on.'''

    big_list = [] # list of lists, where each list is a sentence

    for train_file in training_list:

        f = open(train_file, 'r')
        words = f.readlines() # words is a list, every element being: "word : tag"
        tmp_sentence = [] # list of word, tag pairs (temporary; overwritten after every sentence found)

        for w in words: # w has format: "word : tag"
            
            w = w.split(" : ") # split w into a list with format: w = ["word", "tag"]

            # if the tag has a newline character in it (e.g. "NP0\n"), remove it
            if w[1][-1:] == '\n':
                w[1] = w[1][:-1]

            tmp_sentence.append(w)

            # if the word is "." or "!" or "?" or ";" then it's the end of a sentence, so restart the sentence list
            if w[0] == "." or w[0] == "!" or w[0] == "?" or w[0] == ";":
                big_list.append(tmp_sentence)
                tmp_sentence = []

            # elif words.index(w) == len(words) - 1:
            #     big_list.append(tmp_sentence)
            #     tmp_sentence = []
        
        # if finished the file and still contents in tmp_sentence, append the last "sentence" anyway
        if len(tmp_sentence) != 0:
            big_list.append(tmp_sentence)
            tmp_sentence = []

        f.close()
    
    return big_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    # initialize the training data
    training_data = init_training(training_list)
    for sentence in training_data:
        print(sentence)
        print("")

    print("Starting the tagging process.")
