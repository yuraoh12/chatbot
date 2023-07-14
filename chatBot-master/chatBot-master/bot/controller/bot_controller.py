import pandas as pd
from flask import Blueprint, current_app
from bot.chat.chat_bot import ChatBot

bp = Blueprint('bot', __name__, url_prefix='/bot')

@bp.route('test')
def test():

    dic = {
        "no" : 1,
        "title" : "hihi",
        "body" : "ndfsdf",
        "aaa" : "cv,mcv"
    }

    return "hihi"


@bp.route("question")
def test2():
  bot = ChatBot()
  answer = bot.run_chatbot("안녕")
  result = "{message: "+ answer +"}"
  return result

@bp.route("token")
def test3():
    bot = ChatBot()
    token = bot.get_tokenizer()
    print(token.word_index)

    return "hihihih"


@bp.route("my-train")
def train() :
    bot = ChatBot()
    base_url = current_app.root_path
    df = pd.read_csv(base_url + "/resource/ChatBotData.csv")
    print(df.shape)

    texts = []
    pairs = []
    for text, pair in zip(df['Q'], df['A']):
        texts.append(text)
        pairs.append(pair)


    if load_pickle("questions") == 'f-1':
        questions, answer_in, answer_out = bot.preprocess(texts, pairs)
        save_pickle(questions, "questions")
        save_pickle(answer_in, "answer_in")
        save_pickle(answer_out, "answer_out")

    questions = load_pickle("questions")
    answer_in = load_pickle("answer_in")
    answer_out = load_pickle("answer_out")

    all_sentences = questions + answer_in + answer_out

    bot.fit_texts(all_sentences)

    question_sequences = bot.get_sequences(questions)
    answer_in_sequences = bot.get_sequences(answer_in)
    answer_out_sequences = bot.get_sequences(answer_out)

    padded_questions, padded_answer_in, padded_answer_out = bot.pad_sequences([question_sequences, answer_in_sequences, answer_out_sequences], max_len=30, trun="post", padd="post")

    answer_out_1hot = bot.convert_to_one_hot(padded_answer_out)

    from bot.chat.Seq2seq import Seq2seq
    import numpy as np

    seq2seq = Seq2seq(128, bot.get_vocab_size(), 100, bot.max_length, bot.start_token_idx, bot.end_token_idx)
    seq2seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    for epoch in range(20):
        print(f"processing epoch : {epoch * 10 + 1} ...")
        seq2seq.fit([padded_questions, padded_answer_in], answer_out_1hot, epochs=10, batch_size=16)

        samples = np.random.randint(len(questions), size=3)

        for idx in samples:
            question_inputs = padded_questions[idx]
            results = bot.make_prediction(seq2seq, np.expand_dims(question_inputs, 0))

            results = bot.convert_index_to_text(results, bot.end_token_idx)

    seq2seq.save_weights('sample1')
    return "heheheheh"


import pickle
def save_pickle(obj, name) :
    with open(current_app.root_path + f"/resource/{name}.pkl", 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(name) :
    try :
        with open(current_app.root_path + f"/resource/{name}.pkl", 'rb') as f:
            my_res = pickle.load(f)

        return my_res

    except:
        return "f-1"
