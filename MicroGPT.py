import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass
import time

# hyper parameters
torch.manual_seed(4224)
batch_size = 64
block_size = 256  # maximum context to look at for the next prediction
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
ddp = False  # distributed training

from_checkpoint = True
iteration = 0
checkpoint_path = "checkpoints/"
checkpoint_interval = 100

torch.cuda.empty_cache()

with open("data/fongbe.txt", "r", encoding="utf-8") as f:
    text = f.read()
# text = text[:1500000]
# get a list of characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Char numbers {len(text)}\nvocab size {vocab_size}")

# tokenization
# str -----> int
# we create a mapping ffrom characters to integers
ch2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2ch = {idx: ch for idx, ch in enumerate(chars)}

encode = lambda s: [ch2idx[ch] for ch in s]  # the encoder take a string and return the list of integers
decode = lambda idx_l: ''.join([idx2ch[idx] for idx in idx_l])  # take a list of integers and return the string

# we could use tiktoken (open ai) or sentence piece (google) tokenization

# Convert text encode to torch.Tensor

data = torch.tensor(encode(text), dtype=torch.long)

# split the dataset to train and val set: 90% and 10%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def gelu(x):
    """
    Implementation of the GELU activation from open ai repo
    Reference: https://github.com/openai/gpt-2/blob/master/src/model.py#L25
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def new_softmax(x, dim=-1):
    """
    Softmax used for gpt-2
    """
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    ex = torch.exp(x)
    return ex / torch.sum(ex, dim=dim, keepdim=True)


"""
In order to train our data, we should chunk data, to gain in computation and 
efficiency (chunking and batching). Take data by block
"""


def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))  # we take k random value with size of bloc_size
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits = model(X)

            # make this piece of code as func
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
            # ------#----------

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)  # what i'm looking for ?
        self.key = nn.Linear(n_embed, head_size, bias=False)  # what do i contains
        self.value = nn.Linear(n_embed, head_size, bias=False)  # element that can be aggregate
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        w = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, head_size) @ (B, head_size, T) = (B,T,T)

        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)  # (B,T,C)
        out = w @ v  # (B,T,T) @ (B,T,C) => (B,T,C)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)  # linear projection of outcome
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# The feed forward net, data obtain after attention need to think individually
class FeedForward(nn.Module):
    def __init__(self, n_embed, h_dim=None, dropout=dropout):
        super().__init__()
        h_dim = 4 * n_embed if h_dim is None else h_dim  # the size of ffwd layer
        # self.ln1 = nn.Linear(n_embed, h_dim)
        # self.ln2 = nn.Linear(h_dim, n_embed)
        # self.drop = nn.Dropout(dropout)

        # in order to replicate the last checkpoint
        self.net = nn.Sequential(
            nn.Linear(n_embed, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x = self.ln1(x)
        # x = gelu(x)
        # x = self.ln2(x)
        # x = self.drop(x)
        return self.net(x)


# Create transformer block
class TransformerBlock(nn.Module):
    """ Transformer block"""

    def __init__(self, n_embed, n_head, ffwd_config=None):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        if ffwd_config:
            self.ffwd = FeedForward(n_embed, **ffwd_config)
        else:
            self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # applying residual connection https://arxiv.org/pdf/1512.03385.pdf
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # total number of tokens
    n_embed: int = 1024
    n_head: int = 16
    n_hid: int = 4096
    n_layer: int = 48
    ffwd_dim: int = 8192
    dropout: float = 0.1


@torch.no_grad()
def generate_token(model, idx, max_gen_tokens):
    """Generate new tokens from the current context idx (B, T): the last character
    Not taking history"""
    for _ in range(max_gen_tokens):
        idx_temp = idx if idx.size(1) \
                          <= block_size else idx[:, -block_size:]  # take out from the idx the last
        # block_size token
        logits = model(idx_temp)
        logits = logits[:, -1, :]  # take the last embeddings in each batch (B, C)
        probs = F.softmax(logits, dim=-1)  # get probabilities by applying softmax
        # then get the next idx
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, ffwd_dim, n_layer, dropout=0.1):
        super().__init__()
        self.ffwd_dim = ffwd_dim
        self.dropout = dropout
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_embed, n_head, ffwd_config={'h_dim': ffwd_dim, 'dropout': dropout}) for _ in
             range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # apply weight tying https://paperswithcode.com/method/weight-tying
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = self.drop(tok_emb + pos_emb)  # (B, T, C)
        # apply  attention (B,T,C)
        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = self.ln_f(x)  # final layer normalisation (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size); @TODO: optimize this for inference time
        return logits

    def generate(self, idx, max_gen_tokens):
        return generate_token(self, idx, max_gen_tokens)


# Simple Bigram Language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        # n_embed = C
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T) => (B, T, C) : batch, time to ..., chanel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # apply  attention (B,T,C)
        x = self.ln_f(x)  # final layer normalisation (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # B, T, C = logits.shape
        # logits = logits.view(B * T, C)
        # ----up--- we are not compute loss here anymore
        return logits

    def generate(self, idx, max_gen_tokens):
        """Generate new tokens from the current context idx (B, T): the last character
        Not taking history"""
        return generate_token(self, idx, max_gen_tokens)


def training_loop(model, distributed=False, rank=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters+iteration):

        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

        # sample batch
        xb, yb = get_batch('train')  # but how we ensure all data is learned ?

        if distributed:
            logits = model(xb.to(rank))
            yb = yb.to(rank)
        else:
            logits = model(xb)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        yb = yb.view(B * T)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

        if iter % checkpoint_interval == 0:
            model_name = str(model.__class__.__name__)
            saving_info = {
                "bs": batch_size,
                "blsz": block_size,
                "max_iters": max_iters,
                "lr": learning_rate,
                "n_embd": n_embed,
                "n_head": n_head,
                "n_layer": n_layer,
                "drop": dropout
            }
            if model_name == "GPT":
                saving_info['ffwd_dim'] = model.ffwd_dim
                saving_info['drop'] = model.dropout

            file_name = f"model_at_{iter}_L{str(float('{:.2f}'.format(loss))).replace('.', '_')}"
            torch.save({
                'iteration': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"{checkpoint_path}/{model_name}_{file_name}.pt")
            open(f'{checkpoint_path}/history.txt', 'a', encoding="utf-8").write(
                f"{model_name}_{file_name}: {saving_info} \n"
            )


if from_checkpoint:
    CKPT_PATH = 'checkpoints/model_at_1000_L1_5.pt'

    model = BigramLanguageModel()

    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    iteration = checkpoint['iteration']
    loss = checkpoint['loss']
    print(model, iteration, loss)

else:
    model = GPT(vocab_size, n_embed, n_head, None, n_layer, 0.1)
    model = model.to(device)


def distributed_training(rank, world_size):
    # create the default group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    training_loop(model=ddp_model, distributed=True, rank=rank)
    dist.destroy_process_group()


if __name__ == '__main__':
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    # Create the optimizer

    if ddp:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        world_size = 1
        try:
            mp.spawn(distributed_training,
                     args=(world_size,), nprocs=world_size,
                     join=True)
        except Exception as e:
            raise e
    else:
        training_loop(model=model, distributed=False)

    p_tokens = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_gen_tokens=100)[0].tolist()

    for i in range(0, len(p_tokens), 3):
        print(decode(p_tokens[:i]))
        time.sleep(0.5)

    # PATH = 'checkpoints/model_at_1000_L1_5.pt'
    #
    # net = BigramLanguageModel()
    #
    # checkpoint = torch.load(PATH)
    # net.load_state_dict(checkpoint['model_state_dict'])
    # iteration = checkpoint['iteration']
    # loss = checkpoint['loss']
    # print(net, iteration, loss)

    open('generate.txt', 'w', encoding="utf-8").write(
        decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_gen_tokens=10)[0].tolist()))
