import torch
import json
from MicroGPT import (NGramLanguageModel, GPT,
                      encode, decode, device)

model_name = "BigramLanguageModel_model_at_565_L1_31"


def get_config():
    with open("checkpoints/history.txt", "r") as file:
        history = file.read().splitlines()
        for h in history:
            spl = h.split(':')
            name, config = spl[0], str(':'.join(spl[1:])).replace('\'', '\"')
            if name == model_name:
                print("Config found in history...")
                return json.loads(config)


def generate(write=False):
    model = NGramLanguageModel()

    checkpoint = torch.load(f"checkpoints/{model_name}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(torch.tensor(encode("mi ku do"), dtype=torch.long, device=device).unsqueeze(0))
    output = decode(model.generate(torch.tensor(encode("mi ku do"), dtype=torch.long, device=device).unsqueeze(0),
                                   max_gen_tokens=20)[0].tolist())
    print(output)
    if write:
        open('generate.txt', 'w', encoding="utf-8").write(output)


if __name__ == '__main__':
    generate()
