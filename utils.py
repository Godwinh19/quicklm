import json
import re


def remove_special_characters(batch):
    """ Remove punctuation and lower the text """
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch


def extract_all_chars(batch):
    """ Get all unique characters in text """
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def get_processing_data(dataset):
    dataset = dataset.map(remove_special_characters)

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names["train"]
    )
    vocab_list = list(set(vocabs["train"]["vocab"][0]) |
                      set(vocabs["test"]["vocab"][0])
                      )
    vocab_dict = {v: k for k, v in
                  enumerate(sorted(vocab_list))}  # {'a': 1, 'b': 2, etc}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(f'wav2vec2-large-xlsr-53/vocab.json', 'w', encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return dataset, len(vocab_dict)
