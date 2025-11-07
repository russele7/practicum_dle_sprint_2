import torch
import torch.nn as nn
import re
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import evaluate
import numpy as np


class TextGenerationDataset(Dataset):
    def __init__(self,
                 texts,
                 tokenizer,
                 X_seq_len=15,
                 token_ids_max_length=512):
        self.samples = []

        for line in tqdm(texts):
            line = ' '.join([tokenizer.bos_token, line, tokenizer.eos_token])
            token_ids = tokenizer.encode(line,
                                         add_special_tokens=False,
                                         max_length=token_ids_max_length,
                                         truncation=True)
            if len(token_ids) == 1:
                continue

            for i in range(1, len(token_ids)):

                if i >= X_seq_len:
                    context = token_ids[i-X_seq_len:i]
                else:
                    context = [tokenizer.pad_token_type_id for _ in range(
                        X_seq_len - i)] + token_ids[:i]

                context += [tokenizer.mask_token_id]

                target = token_ids[i]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class RougeDataset(Dataset):
    def __init__(self, texts, n=0.75):
        self.samples = []

        for line in tqdm(texts):
            words = line.split()

            input_words_size = round(len(words) * n)

            input_text = ' '.join(words[:input_words_size])
            output_text = ' '.join(words[input_words_size:])

            if (len(input_text) == 0) or (len(output_text) == 0):
                continue

            self.samples.append((input_text, output_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RnnTextGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128,):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim,
                           hidden_dim,
                           batch_first=True,
                           bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        hidden_state = out[:, -1, :]
        linear_out = self.fc(hidden_state)
        return linear_out


def train_model(model,
                train_loader,
                val_loader,
                val_rouge_dataloader,
                tokenizer,
                optimizer,
                criterion,
                device,
                n_epochs=2,
                max_train_batches_in_epoch=10000,
                max_evalacc_batches_in_epoch=1000,
                max_evalrouge_steps_count=100,):

    train_data = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for i, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
            if i > max_train_batches_in_epoch:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate_loss_accuracy(
            model,
            val_loader,
            criterion,
            device,
            max_evalacc_batches_in_epoch)
        val_rouge1, val_rouge2 = evaluate_rouge(
            model,
            tokenizer,
            val_rouge_dataloader,
            device,
            max_evalrouge_steps_count)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%} "
            f"Val_rouge1 {val_rouge1:.4f} | Val_rouge2 {val_rouge2:.4f}"
        )
        train_data.append(
            {
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_rouge1': val_rouge1,
                'val_rouge2': val_rouge2,
            }
        )
    return train_data


def evaluate_loss_accuracy(
        model,
        loader,
        criterion,
        device,
        max_batches_in_epoch=np.inf):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for i, (x_batch, y_batch) in tqdm(enumerate(loader)):
            if i > max_batches_in_epoch:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            preds = torch.argmax(x_output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            sum_loss += loss.item()
    return sum_loss / len(loader), correct / total


def evaluate_rouge(model,
                   tokenizer,
                   rouge_loader,
                   device,
                   max_steps_count=np.inf):
    model.eval()
    rouge = evaluate.load("rouge")
    rouge1_list = []
    rouge2_list = []

    with torch.no_grad():
        counter = 0
        for x_texts, y_texts in (rouge_loader):
            for j in tqdm(range(len(x_texts))):
                counter += 1
                if counter > max_steps_count:
                    break
                x_text = x_texts[j]
                y_text = y_texts[j]

                predict_text = sentense_generation_inference(
                    x_text,
                    model,
                    tokenizer,
                    device,
                    min_input_seq_len=15,
                    max_total_length=128,
                )

                rouge_data = rouge.compute(
                    predictions=[predict_text], references=[y_text])
                rouge1_list.append(rouge_data['rouge1'])
                rouge2_list.append(rouge_data['rouge2'])

                # if rouge_data['rouge2'] > 0.3:
                #     print(
                #         f'ROUGE1 = {rouge_data["rouge1"]} '
                #         f'ROUGE2 = {rouge_data["rouge2"]}\n'
                #         f'y_text: {y_text}\n'
                #         f'predict_text: {predict_text}\n'
                #     )

    rouge1_average = sum(rouge1_list) / len(rouge1_list)
    rouge2_average = sum(rouge2_list) / len(rouge2_list)
    return rouge1_average, rouge2_average


def word_generation_inference(
    input_text,
    model,
    tokenizer,
    device,
    X_seq_len=15,
):

    line = ' '.join([tokenizer.bos_token, clean_string(input_text)])
    token_ids = tokenizer.encode(
        line, add_special_tokens=False, max_length=512, truncation=True)

    if X_seq_len > len(token_ids):
        token_ids = [tokenizer.pad_token_type_id for _ in range(
            X_seq_len - len(token_ids))] + token_ids
    elif X_seq_len < len(token_ids):
        token_ids = token_ids[len(token_ids) - X_seq_len:]

    token_ids += [tokenizer.mask_token_id]
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
    logit = model(token_ids_tensor)
    pred = torch.argmax(logit, dim=1)
    pred_tok = tokenizer.convert_ids_to_tokens([pred.item()])[0]
    return pred_tok


def sentense_generation_inference(
    input_text,
    model,
    tokenizer,
    device,
    min_input_seq_len=15,
    max_total_length=128,
):

    generated_text = []
    generated_word = ''

    for i in range(max_total_length):
        if max_total_length <= len(input_text) + len(generated_text):
            break
        input_text = ' '.join([input_text, generated_word])

        generated_word = word_generation_inference(
            input_text,
            model,
            tokenizer,
            device,
            min_input_seq_len
        )

        if generated_word == tokenizer.eos_token:
            break

        generated_text.append(generated_word)

    return ' '.join(generated_text)


def evaluate_rouge_gpt(generator,
                       loader,
                       gen_max_length=512,
                       gen_num_return_sequences=1,
                       gen_do_sample=True,
                       gen_top_p=0.95,
                       gen_temperature=0.8,
                       max_steps_cnt=np.inf):
    generator.model.eval()
    rouge = evaluate.load("rouge")
    rouge1_list = []
    rouge2_list = []

    with torch.no_grad():
        counter = 0
        for x_texts, y_texts in loader:
            for j in tqdm(range(len(x_texts))):
                counter += 1
                if counter > max_steps_cnt:
                    break
                x_text = x_texts[j]
                y_text = y_texts[j]

                out = generator(
                    x_text,
                    max_length=gen_max_length,
                    num_return_sequences=gen_num_return_sequences,
                    do_sample=gen_do_sample,      # стохастическая генерация
                    top_p=gen_top_p,          # nucleus sampling
                    temperature=gen_temperature
                )
                predict_text = out[0]["generated_text"][len(x_text)+1:]

                rouge_data = rouge.compute(
                    predictions=[predict_text], references=[y_text])
                rouge1_list.append(rouge_data['rouge1'])
                rouge2_list.append(rouge_data['rouge2'])

                if rouge_data['rouge2'] > 0.3:
                    print(
                        f'ROUGE1 = {rouge_data["rouge1"]} '
                        f'ROUGE2 = {rouge_data["rouge2"]}\n'
                        f'y_text: {y_text}\n'
                        f'predict_text: {predict_text}\n'
                    )

    rouge1_average = sum(rouge1_list) / len(rouge1_list)
    rouge2_average = sum(rouge2_list) / len(rouge2_list)
    return rouge1_average, rouge2_average


def pretty_output(input_text, true_ouput, rnn_generation, gpt_generation):
    return (
        f'INPUT          >> {input_text} <<\n\n'
        f'TRUE OUTPUT    >> {true_ouput} <<\n\n'
        f'RNN GENERATION >> {rnn_generation} <<\n\n'
        f'GPT GENERATION >> {gpt_generation} <<\n\n'
    )


def clean_string(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_rouge_by_tokens(model, tokenizer, loader, device):
    model.eval()
    rouge = evaluate.load("rouge")
    rouge1_list = []
    rouge2_list = []

    with torch.no_grad():
        for i, (x_batch, y_batch) in tqdm(enumerate(loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if i > 100:
                break

            cat_batch = torch.cat(
                [x_batch[:, :-1], y_batch.unsqueeze(1)], dim=1)
            x_seq_size = round(cat_batch.shape[1] * 0.75)
            x_rouge_batch = cat_batch[:, :x_seq_size]
            y_rouge_batch = cat_batch[:, x_seq_size:]
            j = random.randint(0, len(x_batch)-1)
            x_tokens = x_rouge_batch[j]
            y_tokens = y_rouge_batch[j]
            gen_tokens = tokens_generation_inference(
                x_tokens, model, tokenizer,)

            y_str = tensorvector_to_str(y_tokens)
            gen_str = tensorvector_to_str(gen_tokens)

            rouge_data = rouge.compute(
                predictions=[gen_str], references=[y_str])
            rouge1_list.append(rouge_data['rouge1'])
            rouge2_list.append(rouge_data['rouge2'])

    rouge1_average = sum(rouge1_list) / len(rouge1_list)
    rouge2_average = sum(rouge2_list) / len(rouge2_list)
    return rouge1_average, rouge2_average


def token_generation_inference(
    token_ids,
    model,
    device
):
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
    output = model(token_ids_tensor)
    pred = torch.argmax(output, dim=1)

    return pred


def tensorvector_to_str(tensorvector):
    return ' '.join([str(i) for i in tensorvector.tolist()])


def tokens_generation_inference(
    token_ids,
    model,
    tokenizer,
    device,
    max_total_length=128,
):

    for i in range(max_total_length):
        if max_total_length <= len(token_ids) + i:
            break

        new_token = token_generation_inference(
            token_ids,
            model,
            device,
        )
        if new_token == tokenizer.eos_token_id:
            break

        token_ids = torch.cat((token_ids, new_token), dim=0)

    generated_tokens = token_ids[-i:]

    return generated_tokens
