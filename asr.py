# @WIP
"""Here we will train an automatic fon recognition model,
based on our generative language model. We will use
some huggingface resources (https://huggingface.co/) """

import pickle
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForCTC, AutoTokenizer,
                          AutoConfig)
from datasets import DatasetDict, Dataset as TDataset
from utils import get_processing_data
import evaluate

torch.manual_seed(4224)
checkpoint_path = "checkpoints/"
MODEL_CKPT = "facebook/wav2vec2-large-xlsr-53"
model_name = "wav2vec2-large-xlsr-53"
SAMPLING_RATE = 16000
BATCH_SIZE = 8
learning_rate = 1e-4
iteration = 0
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Prepare dataset
with open('data/audio_train.pkl', 'rb') as file:
    train = pickle.load(file)
    train['waveform'] = train['waveform'].values

with open('data/audio_test.pkl', 'rb') as file:
    test = pickle.load(file)
    test['waveform'] = test['waveform'].values

train_dict = TDataset.from_pandas(train)
test_dict = TDataset.from_pandas(test)

data = DatasetDict({"train": train_dict, "test": test_dict})
_, vocab_size = get_processing_data(data)

data = pd.concat([train, test], axis=0, ignore_index=True)

# we only focus on waveform and text columns
data = data[['text', 'waveform']]

max_waveform_len = data['waveform'].apply(len).max()
max_target_len = data['text'].apply(len).max()

cut = int(len(data) * 0.9)
train, test = data.iloc[:cut, :], data.iloc[cut:, :]
train.reset_index(drop=True, inplace=True), test.reset_index(drop=True, inplace=True)


def _config():
    config = AutoConfig.from_pretrained(MODEL_CKPT)

    tokenizer_type = config.model_type if config.tokenizer_class is None else None
    config = config if config.tokenizer_class is not None else None

    return config, tokenizer_type


def _get_tokenizer():
    config, tokenizer_type = _config()
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_CKPT.split('/')[-1]}/",
        config=config,
        tokenizer_type=tokenizer_type,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        truncation=True,
    )
    return tokenizer


class SpeechDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.max_waveform_len = max_waveform_len
        self.max_target_len = max_target_len
        self.tokenizer = _get_tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = torch.tensor(self.data['waveform'][idx])
        transcription = self.data['text'][idx]
        waveform_len = waveform.shape[0]

        if waveform_len < self.max_waveform_len:
            # add padding to normalize waveforms length
            padding_len = self.max_waveform_len - waveform_len
            padding = torch.zeros((1, padding_len)).squeeze(0)
            waveform = torch.cat([waveform, padding], dim=0)
        elif waveform_len > self.max_waveform_len:
            raise Exception("Exceed waveform length")

        target = self.tokenizer(transcription, return_tensors="pt").input_ids.squeeze(0)

        target_len = target.shape[0]
        if target_len < self.max_target_len:
            # Padding the transcription
            padding_len = self.max_target_len - target_len
            padding = torch.tensor([29] * padding_len, dtype=torch.long)
            target = torch.cat([target, padding], dim=0)
        elif target_len > self.max_target_len:
            raise Exception("Exceed target length")

        assert len(waveform) == self.max_waveform_len and len(target) == self.max_target_len
        return waveform, target


dataset, test_dataset = SpeechDataset(train), SpeechDataset(test)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Train

model = AutoModelForCTC.from_pretrained(MODEL_CKPT)
tokenizer = _get_tokenizer()


class SpeechToTextModel(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, vocab_size),
            nn.Dropout(0.2),
        )

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask).logits
        # fine-tuning architecture
        logits = self.net(logits)
        return logits.log_softmax(2)

    def transcribe(self, waveform):
        # waveform normalisation
        # features = torch.log(waveform + 1e-9)
        # input_ids = self.tokenizer(features, return_tensors='pt').input_ids

        # transcription
        logits = self.forward(input_ids=waveform, attention_mask=None)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription


model = SpeechToTextModel(model=model, tokenizer=tokenizer)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CTCLoss()

model.to(device)
for epoch in range(EPOCHS):
    model.train()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids, targets = batch
        logits = model(input_ids=input_ids.to(device), attention_mask=None)
        logits = logits.transpose(0, 1)

        T, B, C = logits.shape  # input_length, batch size, number of class
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
        target_lengths = torch.randint(low=0, high=vocab_size - 1, size=(B,), dtype=torch.long)
        loss = loss_fn(logits, targets.to(device), input_lengths=input_lengths, target_lengths=target_lengths)

        print(f"Loss at epoch {epoch} batch {idx}: {loss.item()}")
        loss.backward()
        optimizer.step()
    saving_info = {
        "bs": BATCH_SIZE,
        "epoch": epoch + iteration,
        "lr": learning_rate,
    }

    # For test purpose
    model.eval()
    for waveform_test_w, waveform_test_t in test_dataloader:
        transcription = model.transcribe(waveform_test_w)
        print("trans: ", transcription, "\n", "actual: ", tokenizer.batch_decode(waveform_test_t)[0])
        break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, f"{checkpoint_path}/{model_name}_{epoch}.pt")
    open(f'{checkpoint_path}/asr_history.txt', 'a', encoding="utf-8").write(
        f"{model_name}_{epoch}: {saving_info} \n"
    )
