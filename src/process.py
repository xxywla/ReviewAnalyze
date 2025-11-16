import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer

import config


def process():
    df = pd.read_csv(config.RAW_DATA_DIR / "online_shopping_10_cats.csv", usecols=['review', 'label'], encoding="utf-8")
    # 过滤数据
    df = df.dropna()

    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'])

    # 构建词表
    JiebaTokenizer.build_vocab(config.VOCAB_FILE, train_data['review'].tolist())
    tokenizer = JiebaTokenizer.from_vocab(config.VOCAB_FILE)

    # 构建训练集
    train_data['review'] = train_data['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))

    # 序列长度 95 分位数
    seq_len_95 = int(train_data['review'].apply(len).quantile(0.95))
    print(f'序列长度 95 分位数: {seq_len_95}')

    train_data.to_json(config.PROCESSED_DATA_DIR / "train_dataset.jsonl", lines=True, orient="records")

    # 构建测试集
    test_data['review'] = test_data['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    test_data.to_json(config.PROCESSED_DATA_DIR / "test_dataset.jsonl", lines=True, orient="records")


if __name__ == '__main__':
    process()
