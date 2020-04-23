import io
import string
from pickle import load
import numpy as np
from numpy import array

from feature_extractor import FeatureExtractor

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
#from nltk.translate.bleu_score import corpus_bleu

class EnglishToHindi:

    def __init__(self,data_path,train_length=2500):
        fe = FeatureExtractor(data_path)
        (self.trainX,self.trainY,self.testX,self.testY,self.eng_tokenizer,self.hindi_tokenizer) = fe.get_train_test_data(train_length)
        self.l = fe.l
        self.eng_vocab_size = fe.eng_vocab_size
        self.hindi_vocab_size = fe.hindi_vocab_size
        self.eng_length = fe.eng_length
        self.hindi_length = fe.hindi_length

    def define_model(self,src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
        model.add(LSTM(n_units))
        model.add(RepeatVector(tar_timesteps))
        model.add(LSTM(n_units, return_sequences=True))
        model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
        return model


    def fit(self,num_epochs):
        # define model
        self.model = self.define_model(self.eng_vocab_size, self.hindi_vocab_size, self.eng_length, self.hindi_length, 256)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        print(self.model.summary())
        #plot_model(model, to_file='model.png', show_shapes=True)
        self.model.fit(self.trainX, self.trainY, epochs=num_epochs, validation_data=(self.testX, self.testY))

    def word_for_id(self,integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                    if index == integer:
                            return word
            return None

    def predict_sequence(self,model, tokenizer, source):
            prediction = model.predict(source, verbose=0)[0]
            integers = [np.argmax(vector) for vector in prediction]
            target = list()
            for i in integers:
                    word = self.word_for_id(i, tokenizer)
                    if word is None:
                            break
                    target.append(word)
            return ' '.join(target)

    def predict(self,source):
        predicted = list()
        for i, source in enumerate(sources):
            source = source.reshape((1, source.shape[0]))
            translation = self.predict_sequence(self.model, self.hindi_tokenizer, source)
            predicted.append(translation.split())
        return predicted

    def evaluate_model(self,sources, raw_dataset):
        actual, predicted = list(), list()
        for i, source in enumerate(sources):
            source = source.reshape((1, source.shape[0]))
            translation = self.predict_sequence(self.model, self.hindi_tokenizer, source)
            raw_src, raw_target = raw_dataset[i]
            if i < 30:
                    print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
            actual.append([raw_target.split()])
            predicted.append(translation)

        # calculate BLEU score
        #print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        return (actual,predicted)

if __name__ == '__main__':
    train_length = 2500
    model = EnglishToHindi('../data/hin.txt')
    model.fit(num_epochs=2)
    model.evaluate_model(model.trainX, model.l[:train_length])
