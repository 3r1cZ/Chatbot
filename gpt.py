import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

# hyperparameters
batch_size = 128 # number of independent sequences being processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # runs on gpu, else cpu
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337) # set seed for reproducibility

 
# reading csv file
text = pd.read_csv("topical_chat.csv", usecols = ["message"])
text = text["message"].tolist()
text = str(text)
# with open('human_chat.txt', 'r', encoding='utf-8') as f:
#    text = f.read()

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

        self.dropout = nn.Dropout(dropout)

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
        wei = self.dropout(wei) # dropping out nodes to prevent overfitting
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs) , aggregated elements
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) , changes output to head_size dimensions
        return out

# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) 
        self.dropout = nn.Dropout(dropout) # dropping out nodes to prevent overfitting

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate outputs from each head
        out = self.dropout(self.proj(out)) # projection: linear transformation of the outcome of the above line
        return out

# a simple linear layer followed by a non-linearity
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # rectifier activation
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Transformer block: communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embd) # computation
        # layer normalization
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections: forking off for computation before returning
        # layer norms applied
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language modelling head

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) , x holds token identities and positions at which they occur
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
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
    
    # outputting generated data from the model
    def output(self, input=None):
        if input is None:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.tensor(encode(input), dtype=torch.long, device=device)[None, ...]
        print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) 

# training the model
def train(model):

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

    m.output()
    print('-------')
    # saving model
    torch.save(m.state_dict(), 'model.pt')


model = GPTLanguageModel()
m = model.to(device) # move model to device for cuda
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)