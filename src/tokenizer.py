import jieba
from tqdm import tqdm

import config


class JiebaTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)

        self.word2index = {word: i for i, word in enumerate(vocab_list)}
        self.index2word = {i: word for i, word in enumerate(vocab_list)}

        self.unk_index = self.word2index[self.unk_token]
        self.pad_index = self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len=None):
        tokens = self.tokenize(text)
        if seq_len is not None:
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens += [self.pad_token] * (seq_len - len(tokens))
        indexes = [self.word2index.get(token, self.unk_index) for token in tokens]
        return indexes

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, "r", encoding='utf-8') as f:
            vocab_list = [line[:-1] for line in f.readlines()]
        return cls(vocab_list)

    @classmethod
    def build_vocab(cls, vocab_path, sentences):
        print('开始构建词表...')
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            for word in jieba.cut(sentence):
                if word.strip() == '':
                    continue
                vocab_set.add(word)
        vocab_list = [cls.pad_token, cls.unk_token] + list(vocab_set)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word in vocab_list:
                f.write(word + "\n")
        print(f'词表保存完成，大小为{len(vocab_list)}')


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "vocab.txt")
    sample_text = "今天天气真好，我们去公园玩吧！"
    encoded = tokenizer.encode(sample_text)
    print(f'原始文本: {sample_text}')
    print(f'编码结果: {encoded}')
