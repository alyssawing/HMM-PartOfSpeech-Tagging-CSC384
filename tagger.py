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
        
        # if finished the file and still contents in tmp_sentence, append the last "sentence" anyway
        if len(tmp_sentence) != 0:
            big_list.append(tmp_sentence)
            tmp_sentence = []

        f.close()
    
    return big_list

def init_I(training_data):
    '''Given a big list of lists (each list is a sentence, each elem in the list
    is a ["word", "tag"] pair), initialize the I matrix, which is row containing
    the probability of each tag for the first word (s0) in the sentence.'''

    all_tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

    first_tags = [] # list of the first tags for the first word of each sentence (evidence for s0)

    for sentence in training_data:
        first_tags.append(sentence[0][1])
    
    print("first tags: ", first_tags)
    first_tags = np.array(first_tags)

    # initialize the I matrix as a numpy list of 0's, the same length as all_tags:
    I = np.zeros(len(all_tags))

    # for each tag in all_tags, count how many times it appears in first_tags and give its probability using numpy
    for i in range(len(all_tags)):
        I[i] = np.count_nonzero(first_tags == all_tags[i]) / len(first_tags) # len(first_tags) is the total number of sentences

    # print(I)
    return I # each index in I corresponds to the index of the tag in all_tags

def init_T(training_data):
    '''Given the big list of lists, initialize the T (state transition) matrix,
    which is an n x n numpy array where n is the number of all tags. Each elem 
    in the matrix is the probability of transition from tag [row] to tag [col].
    In other words, the elem at index ["NPO"]["ADJ"] is the probability of transitioning
    from an NPO tag to an ADJ tag.'''

    all_tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']
    
    # initialize T as a numpy 2D array of 0's, the same size as all_tags:
    T = np.zeros((len(all_tags), len(all_tags)))

    return T

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

    # initialize the I matrix
    init_I(training_data)

    print("Starting the tagging process.")
