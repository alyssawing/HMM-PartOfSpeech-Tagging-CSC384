import os
import sys
import argparse
import numpy as np
import time

# Still TODO: optimize, write outputfile function, account for unknown words

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

def init_I(training_data, all_tags):
    '''Given a big list of lists (each list is a sentence, each elem in the list
    is a ["word", "tag"] pair), initialize the I matrix, which is row containing
    the probability of each tag for the first word (s0) in the sentence.'''

    first_tags = [] # list of the first tags for the first word of each sentence (evidence for s0)

    for sentence in training_data:
        first_tags.append(sentence[0][1])
    
    # print("first tags: ", first_tags)
    first_tags = np.array(first_tags)

    # initialize the I matrix as a numpy list of 0's, the same length as all_tags:
    I = np.zeros(len(all_tags)) 

    # for each tag in all_tags, count how many times it appears in first_tags and give its probability using numpy
    for i in range(len(all_tags)):
        I[i] = np.count_nonzero(first_tags == all_tags[i]) # len(first_tags) is the total number of sentences

    # add a small number to each elem in I to prevent 0 rows, and renormalize
    I = I + 0.01
    I = I / np.sum(I)
    # print(I)
    return I # each index in I corresponds to the index of the tag in all_tags

def init_T(training_data, all_tags):
    '''Given the big list of lists, initialize the T (state transition) matrix,
    which is an n x n numpy array where n is the number of all tags. Each elem 
    in the matrix is the probability of transition from tag [row] to tag [col].
    In other words, the elem at index ["NPO"]["ADJ"] is the probability of transitioning
    from an NPO tag to an ADJ tag.'''
    
    # initialize T as a numpy 2D array of 0.000001's, the same size as all_tags:
    T = np.full((len(all_tags), len(all_tags)), 0.01) # to prevent 0 rows
    # print(len(all_tags))

    for sentence in training_data:
        # count every transition from one tag to another in the sentence.
        # Add 1 to the corresponding elem in T for each transition
        for i in range(len(sentence) - 1):
            T[all_tags.index(sentence[i][1])][all_tags.index(sentence[i+1][1])] += 1
        
    # divide each elem in T by the total number of transitions FROM that tag
    for i in range(len(all_tags)):
        if np.sum(T[i]) != 0: # to prevent dividing 
            T[i] = T[i] / np.sum(T[i])

    return T

def init_M(training_data, all_tags):
    '''Initialize the M (observation probabilities) dictionary, where each key
    is a word that has been encountered in training, and each value is a list 
    (length = 91) of probabilities of each tag for that word. These numbers 
    are normalized by the total number of times that TAG has appeared. P(ek | sk)'''

    M = {}
    tag_count = np.zeros(len(all_tags)) # list of the number of times each tag has appeared

    for sentence in training_data:
        for word in sentence:
            if word[0] not in M:
                M[word[0]] = np.zeros(len(all_tags))                
            M[word[0]][all_tags.index(word[1])] += 1 # add 1 to the corresponding tag (word[1]) in the list at the correct index
            tag_count[all_tags.index(word[1])] += 1 # add 1 to the corresponding tag counter list at the correct index
    
    # Normalize numbers in M by the total number of times that TAG has appeared (not the word)
    for word in M:
        for i in range(len(all_tags)):
            if tag_count[i] != 0:
                M[word][i] = M[word][i] / tag_count[i]

    return M

def init_E(test_file):
    '''Given the test file, create a list of lists of all the sentences and words 
    in the test file.'''
    E = [] # list of lists 
    f = open(test_file, 'r')
    W = list(map(str.rstrip, f.readlines())) # remove the newline characters too. list of all words
    f.close()

    tmp_sentence = [] # list of words in the current sentence
    for w in W:
        if w != "." and w != "!" and w != "?" and w != ";":
            tmp_sentence.append(w)
        else:
            tmp_sentence.append(w)
            E.append(tmp_sentence)
            tmp_sentence = []
    
    # account for last sentence anyway if doesn't end in proper punctuation
    if len(tmp_sentence) != 0:
        E.append(tmp_sentence)

    # for sentence in E:
    #     print(sentence)
    #     print()

    return E

def viterbi(E, S, I, T, M): #TODO - make faster (vectorize?) and account for never seen before words
    '''Execute the Viterbi algorithm to predict the most likely sequence of tags.
    The inputs to the function are:
    - E: list of observations (all words in ONE sentence; test sentence); e.g.  ["she", "walked", "downstairs", "."]
    - S: set of hidden state values (set of all tags)
    - I: initial probabilities row vector; P(s0)
    - T: transition matrix; P(S_k+1 | S_k)
    - M: observation dictionary; P(E_k | S_k)'''

    prob = np.zeros((len(E), len(S)))
    prev = np.zeros((len(E), len(S)))

    # determine values for time step 0 (base case):
    if E[0] not in M.keys():
        # at first, just assume a uniform distribution for all tags (since I should be good enough)
        prob[0] = I*1/len(S) 
        pass
    else:
        # print("base case M[E[0]]: ", M[E[0]])
        prob[0] = I*M[E[0]]

    prev[0] = None
    x = np.argmax(prob[0])

    # create distribution (array with 91 elems) for unseen words;
    unknown_tagp = np.zeros(len(S))
    unknown_tagp[S.index("NP0")] = 0.1 # proper nouns
    unknown_tagp[S.index("AJ0")] = 0.1 # adjectives
    unknown_tagp[S.index("NN2")] = 0.1 # plural common nouns
    unknown_tagp[S.index("NN1")] = 0.1 # singular common nouns
    # make the remaining values equal so it adds to 1:
    unknown_tagp[unknown_tagp == 0] = 0.6 / (len(S) - 2)

    # time steps 1 to length(E) (recursive case):
    for t in range(1, len(E)): # t is the index of the current word
        for i in range(0, len(S)): # i is the index of the current tag

            # if it's a word that hasn't been seen before: TODO. Also TODO - deal with tags never appearing (don't make 0)
            if E[t] not in M.keys():
                # print("word not in M: ", E[t])
                #TODO - set prob[t][i] and prev[t][i] to something
                x = np.argmax(prob[t-1]*T.T[i]*unknown_tagp[i]) # or * 1/len(S))
                prob[t][i] = prob[t-1][x] * T[x][i] * unknown_tagp[i] # the probability of the current tag given the previous tag
                prev[t][i] = x # the previous tag that maximizes the probability

            else: 
                x = np.argmax(prob[t-1]*T.T[i]*M[E[t]][i])
                prob[t][i] = prob[t-1][x] * T[x][i] * M[E[t]][i] # the probability of the current tag given the previous tag
                prev[t][i] = x # the previous tag that maximizes the probability
        
        # normalize to prevent probabilities from getting too small:
        if np.sum(prob[t]) != 0:
            prob[t] = prob[t] / np.sum(prob[t])
        else:
            print("PROBLEM: all of prob[t] is 0 for E[t] = ", E[t]) # TODO - fix this
            pass

    # for l in prev:
    #     print(l)
    #     print()

    tag_sequence = []

    # Backtrack to find the most likely sequence of tags:
    x = np.argmax(prob[len(E)-1]) # in the last row of prob, get the index of the max probability:
    # in the last row of prev, get the tag corresponding to that maxp index:
    tag_sequence = np.concatenate((int(x), tag_sequence), axis=None)

    # now backtrack through prev to get the rest of the tags:
    for t in range(len(E)-1, 0, -1): # go backwards from the last to the first row of prev
        tag_sequence = np.concatenate((prev[t][int(x)], tag_sequence), axis=None)
        x = prev[t][int(x)]

    # print("tag_sequence: ", tag_sequence)
    # show tags corresponding to each tag in tag sequence:
    # for i in range(len(tag_sequence)):
        # print(S[int(tag_sequence[i])])

    return tag_sequence

def get_accuracy(tag_guesses, answerfile, all_tags):
    '''After running Viterbi, compare the predicted tag sequence to the actual
    tag sequence in the test file. Return the accuracy. The inputs are:
    - tag_guesses: list of lists of the predicted tags for each sentence in the test file
    - answerfile: the test file with the answers
    - all_tags: list of all possible tags'''

    f = open(answerfile, 'r')
    W = list(map(str.rstrip, f.readlines())) # remove the newline characters too. list of all words
    f.close()

    # get the true tag sequence:
    true_tag_sequence = []
    f = open(answerfile, 'r')
    words = f.readlines() # words is a list, every element being: "word : tag"
    tmp_sentence = [] # list of word, tag pairs (temporary; overwritten after every sentence found)

    for w in words: # w has format: "word : tag"
        w = w.split(" : ") # split w into a list with format: w = ["word", "tag"]

        # if there's no index error and has a newline character in it (e.g. "NP0\n"), remove it
        if w[1] != '' and w[1][-1:] == '\n':
            w[1] = w[1][:-1]
        tmp_sentence.append(w)

        # if the word is "." or "!" or "?" or ";" then it's the end of a sentence, so restart the sentence list
        if w[0] == "." or w[0] == "!" or w[0] == "?" or w[0] == ";":
            true_tag_sequence.append(tmp_sentence)
            tmp_sentence = []
    
    # if finished the file and still contents in tmp_sentence, append the last "sentence" anyway
    if len(tmp_sentence) != 0:
        true_tag_sequence.append(tmp_sentence) 
        tmp_sentence = []
    f.close()
    
    # format of true_tag_sequence: list of lists of word, tag pairs
    
    # for sentence in true_tag_sequence:
    #     print(sentence)
    #     print()

    # for every sentence and every tag, compare and get accuracy:
    correct = 0
    total = 0
    for i in range(len(true_tag_sequence)):
        for j in range(len(true_tag_sequence[i])):
            total += 1
            # compare the actual tag to the predicted tag:
            # print(tag_guesses)
            if true_tag_sequence[i][j][1] == all_tags[int(tag_guesses[i][j])]:
                correct += 1
            else:
                print("wrong guess for this word: ", true_tag_sequence[i][j][0], "expected tag: ", true_tag_sequence[i][j][1])

    return correct/total

def write_output(tag_guesses, E, outputfile, all_tags):
    '''Write the predicted tag sequence to a file. The inputs are:
    - tag_guesses: list of lists of the predicted tags for each sentence in the test file
    - E: the test list of lists (just the words to add the tags to with format word : tag)
    - outputfile: the file to write the output to with formaat word : tag on each line
    - all_tags: list of all possible tags'''

    # Read through every word (line) in the test file and add the : tag from the tag_guesses list:
    # f = open(testfile, 'r')
    # testwords = f.readlines() # words is a list, every element being: "word : tag"
    # # remove the newline characters too:
    # testwords = list(map(str.rstrip, testwords))
    # f.close()

    # Write in the output file the word corresponding to the tag:
    f = open(outputfile, 'w')
    for sentence_i in range(len(tag_guesses)):
        for word_i in range(len(tag_guesses[sentence_i])):
            # hardcode to ensure all punctuation correct:
            if E[sentence_i][word_i] == "." or E[sentence_i][word_i] == "!" or E[sentence_i][word_i] == "?" \
                or E[sentence_i][word_i] == ";" or E[sentence_i][word_i] == "," or E[sentence_i][word_i] == ":":
                # print("replaced punctuation")
                f.write(E[sentence_i][word_i] + " : " + "PUN" + "\n")
            elif E[sentence_i][word_i] == '"':
                # print("replaced quotation")
                f.write(E[sentence_i][word_i] + " : " + "PUQ" + "\n")
            else:
                f.write(E[sentence_i][word_i] + " : " + all_tags[int(tag_guesses[sentence_i][word_i])] + "\n")

    f.close()

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

    time1 = time.time()

    print("************************* INITIALIZING DATA ************************")

    all_tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

    # initialize the training data
    training_data = init_training(training_list)

    # training_data = training_data[0:2] # shorten training to 2 sentences for testing
    # for sentence in training_data:
    #     print(sentence)
    #     print("")
    # initialize the 3 probability matrices
    I = init_I(training_data, all_tags)
    # print(I)
    T = init_T(training_data, all_tags)
    # for line in T:
    #     print(line)
    M = init_M(training_data, all_tags)
    # print(M)

    # initialize the other inputs to viterbi:
    E = init_E(args.testfile)

    print("*************************** TAGGING ********************************")

    tag_guesses = []
    for i in range(len(E)): # for every sentence
        tag_guesses.append(viterbi(E[i], all_tags, I, T, M))
    
    # answerfile = training_list[0] #  for now, test on the same file that was used to train
    answerfile = "training1.txt" # for testing on a specific dataset - MODIFY depending on what answers are

    # write to output file:
    write_output(tag_guesses, E, args.outputfile, all_tags)

    time2 = time.time()
    print("time taken is {}\n".format(time2-time1))
    print("accuracy is {}".format(get_accuracy(tag_guesses, answerfile, all_tags))) #TODO  - comment out later