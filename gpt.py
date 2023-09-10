import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # number of independent sequences being processed in parallel
block_size = 8 # the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # runs on gpu, else cpu
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337) # set seed for reproducibility

with open('human_chat.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) } # characters to integers
itos = { i:ch for i,ch in enumerate(chars) } # integers to characters
encode = lambda s: [stoi[c] for c in s] # encoder: receive a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: receive a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be for training, rest val
train_data = data[:n]
val_data = data[n:]

# data loading by batch
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # move data to device for cuda
    return x, y

# averages loss over eval_iters batches of training
@torch.no_grad() # stop calling backward for efficiency
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# one head of self-attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix

    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities") | weight calculation providing averages for past and current tokens in a data-dependent way
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # matrix multiplication: (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) , triangular masking
        wei = F.softmax(wei, dim=-1) # (B, T, T) , exponentiates and normalizes in order to create the weighting (-inf will turn into 0)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs) , aggregated elements
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) , changes output to head_size dimensions
        return out

# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate outputs from each head
        return out

# a simple linear layer followed by a non-linearity
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(), # rectifier activation
        )

    def forward(self, x):
        return self.net(x)

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # language modelling head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) , x holds token identities and positions at which they occur
        x = self.sa_heads(x) # apply one head of self-attention (B,T,C)
        x = self.ffwd(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # batch, time, channels
            # adjusting to match cross_entropy
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # checking quality of predictions

        return logits, loss

    # extends each batch (B) in the time (T) dimension for max_new_tokens
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # adjust idx to fix last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device) # move model to device for cuda

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zeroing all the gradients from previous step
    loss.backward() # getting the gradients from all of the parameters
    optimizer.step() # using gradients to update parameters

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))