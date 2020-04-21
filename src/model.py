import io
import string
from pickle import load
from numpy import array

from feature_extrator import FeatureExtractor

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

class EnglishToHindi:

    def __init__(self,data_path):
        fe = FeatureExtractor(data_path)
        (self.trainX,self.trainY,self.testX,self.testY,self.eng_tokenizer,self.hindi_tokenizer) = fe.get_train_test_data()
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


    def fit(self):
        # define model
        model = self.define_model(self.eng_vocab_size, self.hindi_vocab_size, self.eng_length, self.hindi_length, 256)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        print(model.summary())
        #plot_model(model, to_file='model.png', show_shapes=True)
        model.fit(self.trainX, self.trainY, epochs=150, validation_data=(self.testX, self.testY))

if __name__ == '__main__':
    model = EnglishToHindi('../data/hin.txt')
    model.fit()
