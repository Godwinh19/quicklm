import torch
import json
from MicroGPT import (BigramLanguageModel, GPT,
                      decode, device)

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


def generate():
    model = BigramLanguageModel()

    checkpoint = torch.load(f"checkpoints/{model_name}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    open('generate.txt', 'w', encoding="utf-8").write(
        decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_gen_tokens=1000)[0].tolist()))

    return model


if __name__ == '__main__':
    generate()
