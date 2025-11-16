import torch

from src import config
from src.model import ReviewAnalyzeModel
from src.tokenizer import JiebaTokenizer


def predict_batch(input_tensors, model):
    model.eval()
    with torch.no_grad():
        # output.shape is [batch_size]
        output = model(input_tensors)
        return torch.sigmoid(output).tolist()


def predict(text, model, tokenizer, device):
    input_index_list = tokenizer.encode(text)
    input_tensor = torch.tensor([input_index_list]).to(device)
    batch_result = predict_batch(input_tensor, model)
    return batch_result[0]


def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(config.VOCAB_FILE)
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_index).to(device)
    model.load_state_dict(torch.load(config.MODEL_FILE_NAME))

    print('请输入一段评论 q 或 quit 退出')
    while True:
        cur_input = input('> ')
        if cur_input == 'q' or cur_input == 'quit':
            break
        if cur_input.strip() == '':
            continue
        prob = predict(cur_input, model, tokenizer, device)
        if prob >= 0.5:
            print(f'正面评论，置信度: {prob:.4f}')
        else:
            print(f'负面评论，置信度: {1 - prob:.4f}')


if __name__ == '__main__':
    run_predict()
