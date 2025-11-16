import torch

from src.dataset import get_dataloader
import config
from predict import predict_batch
from src.model import ReviewAnalyzeModel
from src.tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device):
    total_num, acc_num = 0, 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)

        predicts = predict_batch(inputs, model)
        for predict, target in zip(predicts, targets.tolist()):
            pred_label = 1 if predict > 0.5 else 0
            if pred_label == target:
                acc_num += 1
            total_num += 1
    return acc_num / total_num


def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader(False)

    tokenizer = JiebaTokenizer.from_vocab(config.VOCAB_FILE)

    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_index).to(device)
    model.load_state_dict(torch.load(config.MODEL_FILE_NAME))

    accuracy = evaluate(model, dataloader, device)

    print(f'准确率: {accuracy:.4f}')


if __name__ == '__main__':
    run_evaluate()
