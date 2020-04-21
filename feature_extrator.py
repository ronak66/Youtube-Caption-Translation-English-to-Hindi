import io
import string
import numpy as np
from numpy import array

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class FeatureExtractor:

    def __init__(self,data_path):
        #data_path = "hin.txt"
        self.lines = io.open(data_path, encoding = "utf-8").read().split("\n")
        self.lines  = self.lines[:-1]
        self.lines = [line.split("\t") for line in self.lines]

    def clean_data(self):
        cleaned = list()
        for pair in self.lines:
            clean_pair = list()
            for line in pair:
                line.split()
                line = [word.lower() for word in line]
                clean_pair.append(''.join(line))
            cleaned.append(clean_pair)
        self.lines = np.array(cleaned)
        table = str.maketrans('', '', string.punctuation)
        self.l = [[w[0].translate(table), w[1].translate(table)] for w in self.lines]
        self.l = np.array(self.l)
        return

    def create_tokenizer(self,lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def max_length(self,lines):
        return max(len(line.split()) for line in lines)

    def encode_sequences(self,tokenizer, length, lines):
        X = tokenizer.texts_to_sequences(lines)
        X = pad_sequences(X, maxlen=length, padding='post')
        return X

    def encode_output(self,sequences, vocab_size):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y

    def get_train_test_data(self,train_length = 2500):
        self.clean_data()
        # prepare english tokenizer
        eng_tokenizer = self.create_tokenizer(self.l[:, 0])
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        eng_length = self.max_length(self.l[:, 0])
        print('English Vocabulary Size: %d' % eng_vocab_size)
        print('English Max Length: %d' % (eng_length))
        # prepare german tokenizer
        hindi_tokenizer = self.create_tokenizer(self.l[:, 1])
        hindi_vocab_size = len(hindi_tokenizer.word_index) + 1
        hindi_length = self.max_length(self.l[:, 1])
        print('Hindi Vocabulary Size: %d' % hindi_vocab_size)
        print('Hindi Max Length: %d' % (hindi_length))
        trainX = self.encode_sequences(eng_tokenizer, eng_length, self.l[:train_length][:, 0])
        trainY = self.encode_sequences(hindi_tokenizer, hindi_length, self.l[:train_length][:, 1])
        trainY = self.encode_output(trainY, hindi_vocab_size)
        # prepare validation data
        testX = self.encode_sequences(eng_tokenizer, eng_length, self.l[train_length:][:, 0])
        testY = self.encode_sequences(hindi_tokenizer, hindi_length, self.l[train_length:][:, 1])
        testY = self.encode_output(testY, hindi_vocab_size)
        return (trainX,trainY,testX,testY,eng_tokenizer,hindi_tokenizer)
    

if __name__ == '__main__':
    fe = FeatureExtractor('hin.txt')
    fe.get_train_test_data()

