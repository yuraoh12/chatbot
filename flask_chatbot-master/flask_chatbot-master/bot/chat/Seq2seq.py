import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
## 모델 객체
class Encoder(tf.keras.Model):
  def __init__(self, units, vocab_size, embedding_dim, time_steps):
    super(Encoder, self).__init__()

    ## Embedding -> 카테고리컬 단어값을 고차원으로 바꾸는 것(우리는 원핫을 사용함)
    self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps) ## 단어 개수, 변환하고자 하는 임베딩 차원, 한 문장의 길이
    self.dropout = Dropout(0.2) ## 과적합을 방지하기 위한 하이퍼파라미터. 임의로 20% 뉴런을 잡아서 비활성화 시킴
    self.lstm = LSTM(units, return_state=True) ## 최종 히든 스테이트를 얻어야 벡터 콘텍스트에 넣을 수 있음.

  def call(self, inputs):
    print(inputs)
    x = self.embedding(inputs) ## 임베딩 세팅
    x = self.dropout(x) ## 과적합 방지 파라미터 세팅
    x, hidden_state, cell_state =self.lstm(x) ## 정답, 히든 스테이트, 셀 스테이트

    return [hidden_state, cell_state]

class Decoder(tf.keras.Model):
  def __init__(self, units, vocab_size, embedding_dim, time_steps):
    super(Decoder, self).__init__()
    self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps)
    self.dropout = Dropout(0.2)
    self.lstm = LSTM(units,
                     return_state=True, # 스테이트값을 알아야 다음 셀에서 진행 가능
                     return_sequences=True ## 각 유닛의 스테이트값을 다 얻어서 결과를 얻어야 하므로
    )
    self.dense = Dense(vocab_size, activation='softmax') ## 결과를 얻기 위한 출력층

  def call(self, inputs, initial_state): ## initial_state는 encoder의 출력값
    x = self.embedding(inputs)
    x = self.dropout(x)
    x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state)
    x = self.dense(x) # 최종 결과값은 출력층을 거쳐 결과를 낸다

    return x, hidden_state, cell_state

class Seq2seq(tf.keras.Model):
  def __init__(self, units, vocab_size, embedding_dim, time_steps, start_token, end_token):
    super(Seq2seq, self).__init__()

    self.start_token = start_token
    self.end_token = end_token
    self.time_steps = time_steps # 문장의 길이(30)

    self.encoder = Encoder(units, vocab_size, embedding_dim, time_steps)
    self.decoder = Decoder(units, vocab_size, embedding_dim, time_steps)

  def call(self, inputs, training=True): ## training: 학습용, 예측용 구별
    if training: ## 학습일 땐,
      encoder_inputs, decoder_inputs = inputs ## 인코더, 디코더 모두 동일한 입력값 넣는다.
      context_vector = self.encoder(encoder_inputs) ## 인코더에 넣어서 벡터 얻어냄
      decoder_outputs, _, _ = self.decoder(inputs=decoder_inputs, initial_state=context_vector) ## 얻어낸 인코더의 벡터값을 디코더에 사용

      return decoder_outputs

    else: ## 예측일 땐,
      context_vector = self.encoder(inputs) ##
      target_seq = tf.constant([[self.start_token]], dtype=tf.float32) ## 첫번째는 무조건 <START>,
      results = tf.TensorArray(tf.int32, self.time_steps) ## 결과 배열. 그래프 그리기 위해 텐서 배열로 담는다.

      for i in tf.range(self.time_steps):
        decoder_output, decoder_hidden, decoder_cell = self.decoder(target_seq, initial_state=context_vector)
        decoder_output = tf.cast(tf.argmax(decoder_output, axis=-1), dtype=tf.int32)
        decoder_output = tf.reshape(decoder_output, shape=(1, 1))
        results = results.write(i, decoder_output)

        if decoder_output == self.end_token:
          break

        target_seq = decoder_output
        context_vector = [decoder_hidden, decoder_cell]

      return tf.reshape(results.stack(), shape=(1, self.time_steps))