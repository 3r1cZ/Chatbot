import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # number of independent sequences being processed in parallel
block_size = 8 # the maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # runs on gpu, else cpu
eval_iters = 200
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

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
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