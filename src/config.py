from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VOCAB_FILE = PROCESSED_DATA_DIR / "vocab.txt"

MODEL_PATH = Path(__file__).parent.parent / "model"
MODEL_FILE_NAME = MODEL_PATH / "model.pt"

LOGS_DIR = Path(__file__).parent.parent / "logs"

DATASET_SAMPLE_RATE = 0.1

SEQ_LEN = 128
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 5

if __name__ == '__main__':
    print(DATA_DIR)
