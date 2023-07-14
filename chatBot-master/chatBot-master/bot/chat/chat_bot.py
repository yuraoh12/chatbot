from bot.chat.Seq2seq import Seq2seq
from bot.chat import constants as co
import re
import numpy as np
from joblib import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from flask import current_app
import warnings

warnings.filterwarnings('ignore')

class ChatBot():

    def __init__(self):
        self.okt = Okt()
        self.max_length = 30
        self.padding_size = self.max_length
        self.tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
        self.num_of_epochs = 20
        self.batch_size = 16

    def load_test(self):
        base_dir = current_app.root_path
        name = '/resource/tokenizer'
        self.tokenizer = load(base_dir + name)

        self.DATA_LENGTH = co.DATA_LENGTH
        START_TOKEN = self.tokenizer.word_index['<START>']
        END_TOKEN = self.tokenizer.word_index['<END>']

        self.bot = Seq2seq(co.UNITS, co.VOCAB_SIZE, co.EMBEDDING_DIM, co.TIME_STEPS, START_TOKEN, END_TOKEN)
        self.bot.load_weights(base_dir + '/resource/test')

    def get_vocab_size(self):
        return len(self.tokenizer.index_word) + 1
    def get_tokenizer(self):
        return self.tokenizer

    def get_bot(self):
        return self.bot

    def fit_texts(self, sentences):
        self.tokenizer.fit_on_texts(sentences)
        self.start_token_idx = self.tokenizer.word_index['<START>']
        self.end_token_idx = self.tokenizer.word_index['<END>']

    def get_sequences(self, sentences):
        sequences = self.tokenizer.texts_to_sequences(sentences)
        return sequences

    ## 모델이 예측한 인코딩된 값을 다시 문자로 디코딩 해주는 함수
    def convert_index_to_text(self, indexes, end_token):
        sentence = ''
        for index in indexes:  ## 문장의 순서
            if index == end_token:  ## 문장의 마지막이면 종료
                break
            if index > 0 and self.tokenizer.index_word[index] is not None:  ## 단아 사전에 존재하고 올바른 인덱스라면
                sentence += self.tokenizer.index_word[index]  # 최종 문자열에 이어 붙인다.
            else:
                sentence += ''  # 없는 거면 공백.

            sentence += ' '  # 한 형태소가 끝나면 띄어쓰기
        return sentence

    def run_chatbot(self, question):
        question_inputs = self.make_question(question)
        results = self.make_prediction(self.bot, question_inputs)
        print(results)
        results = self.convert_index_to_text(results, '<END>')
        return results

    def make_question(self, sentence):
        sentence = self.clean_and_morph(sentence)
        question_sequence = self.tokenizer.texts_to_sequences([sentence])
        print(question_sequence)
        question_padded = pad_sequences(question_sequence, maxlen=co.MAX_LENGTH, truncating=co.TRUNCATING, padding=co.PADDING)

        return question_padded

    def clean_and_morph(self, sentence, is_question=True):
        ## 한글만 남기기
        sentence = self.clean_sentence(sentence)

        ## 형태소로 쪼개기
        sentence = self.process_morph(sentence)

        if is_question:
            return sentence

        else:
            ## 후에 토크나이저하기 위해서는 공백이 꼭 들어가야 함.
            return ('<START> ' + sentence, sentence + ' <END>')

    def make_prediction(self, model, question_inputs):
        results = model(inputs=question_inputs, training=False)
        results = np.asarray(results).reshape(-1)
        return results

    def process_morph(self, sentence):
      return ' '.join(self.okt.morphs(sentence))

    def clean_sentence(self, sentence):
        sentence = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]', r'', sentence)
        return sentence

    def preprocess(self, texts, pairs):
        questions = []
        answer_in = []
        answer_out = []

        ## 질문에 대한 전처리
        for text in texts:
            question = self.clean_and_morph(text, is_question=True)
            questions.append(question)

        ## 답변에 대한 전처리
        for pair in pairs:
            in_, out_ = self.clean_and_morph(pair, is_question=False)
            answer_in.append(in_)
            answer_out.append(out_)

        return questions, answer_in, answer_out

    def pad_sequences(self, seq_list, max_len, trun, padd):
        return list(map(
            lambda x : pad_sequences(x, maxlen=max_len, truncating=trun, padding=padd),
            seq_list
        ))

    def convert_to_one_hot(self, padded):
        one_hot_vector = np.zeros((len(padded), self.padding_size, self.get_vocab_size()))


        for i, sequence in enumerate(padded):
            for j, index in enumerate(sequence):
                one_hot_vector[i, j, index] = 1

        return one_hot_vector