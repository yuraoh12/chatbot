VOCAB_SIZE = 2301 # 총 고유 단어의 개수
MAX_LENGTH = 30 # 단어의 최대 개수
BUFFER_SIZE = 1000 # 버퍼 크기??
BATCH_SIZE = 16 # 배치 크기??
EMBEDDING_DIM = 100 # 임베딩 차원??
TIME_STEPS = MAX_LENGTH # 전이 횟수
UNITS = 128   # 입력값의 개수??
DATA_LENGTH = 1000 # 학습 문장의 개수
SAMPLE_SIZE = 3 # ??
NUM_EPOCHS = 10 # 학습 반복 횟수
TRUNCATING='post' # 최대 개수 넘어갈 때 뒤에서 자름
PADDING='post' # 최대 개수보다 모자랄때 뒤에서 0으로 채움

