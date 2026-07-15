---
title: "Building MiniParT"
teaching: 30
exercises: 15
---

:::::: questions
- What are the four main pieces that make up the MiniParT model, and what does each one do?
- What is self-attention, and why does it matter that the two jets can "look at" each other?
- Why does the model use mean pooling to go from two jet descriptions down to one event summary?
- Why not simply concatenate the two jets' features together instead of using a transformer?
- What does data actually look like, shape by shape, as it flows through the model from input to final prediction?
::::::

:::::: objectives
- Identify the four building blocks of `MiniParT`: embedding, self-attention, pooling, and classification head.
- Explain what an embedding layer does to the raw 10-number jet description.
- Explain self-attention in terms of the two jets in this problem exchanging information about each other.
- Explain why averaging (mean pooling) is used to combine the two jets' descriptions into one.
- Trace the shape of a batch of data through every layer of the forward pass.
::::::

## Run this first

```python
class MiniParT(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, hidden_dim=128, num_classes=3):
        super(MiniParT, self).__init__()
        
        # 1. Linear projection (Embedding)
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 2. Transformer Encoder Layer (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True,
            dropout=0.1
        )
        # Using just 2 layers for a "mini" model to keep local training fast
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Classification Head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len=2, Features)
        
        # Project features
        x = self.embedding(x) # shape: (Batch, 2, embed_dim)
        
        # Apply self-attention
        x = self.transformer(x) # shape: (Batch, 2, embed_dim)
        
        # Mean pooling over the sequence (the 2 jets)
        x_pooled = x.mean(dim=1) # shape: (Batch, embed_dim)
        
        # Classify
        out = self.mlp(x_pooled) # shape: (Batch, num_classes)
        return out

torch.manual_seed(42)
np.random.seed(42)

model = MiniParT(input_dim=len(FEATURE_NAMES))
print(model)
```

```output
MiniParT(
  (embedding): Linear(in_features=10, out_features=64, bias=True)
  (transformer): TransformerEncoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
        )
        (linear1): Linear(in_features=64, out_features=128, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=128, out_features=64, bias=True)
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (mlp): Sequential(
    (0): Linear(in_features=64, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=128, out_features=3, bias=True)
  )
)
```

(Exact formatting of the printed module tree can vary slightly by
PyTorch version, but the layers and shapes will match.) Instantiating
`MiniParT` and calling `print(model)` prints a summary of every layer
defined in the class, in order - a useful sanity check that the
architecture you're about to read about below is the architecture
PyTorch actually built.

---
*Run the block above first, then read on to see what each part does.*

This is the heart of the project: `class MiniParT`, defined above.
Nothing here is magic - every line is one of a handful of building
blocks stacked together, walked through piece by piece below.

```python
class MiniParT(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, hidden_dim=128, num_classes=3):
```

First, the knobs (**hyperparameters** - settings *we* choose, as opposed
to numbers the model learns on its own):

- **`input_dim`** - how many numbers describe one jet: `len(FEATURE_NAMES) = 10`.
- **`embed_dim=64`** - the size of the model's own "internal language."
- **`num_heads=4`** - how many independent "attention" viewpoints the model uses at once.
- **`hidden_dim=128`** - the size of an internal working layer used inside the transformer and the final decision step.
- **`num_classes=3`** - how many possible answers there are (Hbb, Hcc, QCD).

## Fixing the random seed, before building anything

Conceptually, before thinking about any of the pieces below, fix the
random seed. In the code above this line sits right before `model =
MiniParT(...)`, since defining the class itself consumes no randomness -
only *instantiating* it does, by initializing every layer's starting
weights:

```python
torch.manual_seed(42)
np.random.seed(42)
```

::: callout
`torch.manual_seed(42)` and `np.random.seed(42)` fix the starting point
for every random process used from here on - the model's initial
weights, dropout, and batch shuffling. Without a fixed seed, every
learner (and every re-run) trains a slightly different model, so
nobody's numbers will exactly match anyone else's, including the example
numbers quoted later in this lesson. Run this cell before building or
training the model, not after.

The `42` itself isn't special - it's just an arbitrary fixed integer.
Any number would work equally well as a seed; what matters is picking
*some* fixed value so the "random" numbers PyTorch and NumPy generate
follow the same sequence every time the code runs, the same way
`random_state=42` did for the train/test split in
[Preparing the Data](05-preparing-the-data.md).
:::

## Piece 1 - The embedding layer: translating raw numbers into a richer language

```python
self.embedding = nn.Linear(input_dim, embed_dim)
```

Each jet arrives as just 10 plain numbers - a cramped way to represent
something as complex as a jet. `nn.Linear(10, 64)` is a learnable
transformation that re-expresses those 10 numbers as 64, by learning a
useful *combination* of the originals that's more expressive to work with
internally. This step is called an **embedding**.

## Piece 2 - Self-attention: letting the two jets "talk" to each other

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=hidden_dim,
    batch_first=True,
    dropout=0.1,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

This is the "Transformer" in MiniParT. Its core trick is
**self-attention**: every jet in the pair looks at every other jet
(including itself) and decides how much to pay attention to it before
updating its own internal description. With only 2 jets, jet A "looks at"
jet B and asks "given what you look like, how should I adjust what I
think I am?" - and jet B does the same. This matters physically: whether
a jet pair is Hbb, Hcc, or QCD isn't just about what one jet looks like
alone, it's about the *relationship* between the two. Self-attention lets
the model reason about jets *together* instead of scoring each one
separately.

::: callout
This is the same core mechanism behind large language models like GPT and
Claude - there, self-attention lets each word in a sentence "look at"
every other word to figure out how they relate before predicting what
comes next. Here, it's the same math applied to 2 jets instead of a
sequence of text tokens.
:::

A few of the settings:

- **`nhead=num_heads=4`** - the model computes attention 4 different ways
  in parallel (**attention heads**), like 4 reviewers each free to pick up
  on a different kind of relationship between the jets, then combined.
- **`dim_feedforward=hidden_dim=128`** - after attention, a small extra
  processing step further refines each jet's description; `128` is how
  wide that step is.
- **`dropout=0.1`** - during training, randomly "switches off" 10% of
  internal connections on every pass. This forces the model not to
  over-rely on any single connection, helping it generalize instead of
  memorize. (Turned off automatically when just making predictions.)
- **`num_layers=2`** - two self-attention layers stacked. After the first,
  each jet's description already includes information "borrowed" from the
  other; a second layer lets that refined information get exchanged again.

Why self-attention instead of just concatenating the two jets' 10
features into one 20-number vector? Concatenation would force you to
arbitrarily label one jet "first" and the other "second," and the
network's answer could depend on that meaningless order. Self-attention
followed by mean pooling treats the two jets symmetrically: swapping
which is listed first doesn't change the answer, which matches the
physics - a jet pair "is" Hbb regardless of which jet was listed first.

## Piece 3 - Mean pooling: turning "2 jets" into "1 decision"

```python
x_pooled = x.mean(dim=1)   # shape: (Batch, embed_dim)
```

After the transformer, we still have a separate description for each of
the 2 jets, but we need one answer per *event*. **Mean pooling** averages
the two jets' 64-number descriptions into a single 64-number summary. It's
a deliberately simple way to combine them - more sophisticated models use
fancier methods, but averaging is easy to understand and works fine here.

## Piece 4 - The classification head: making the final call

```python
self.mlp = nn.Sequential(
    nn.Linear(embed_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, num_classes),
)
```

A small standard neural network (MLP) that takes the pooled 64-number
summary and turns it into 3 numbers - one raw score per class. Higher
score means the model thinks that class is more likely.

- `nn.Linear(64, 128)` expands the summary into a wider working space.
- `nn.ReLU()` is an **activation function**: zeroes out negative numbers,
  leaves positive ones unchanged. Without it, stacking linear layers
  would mathematically collapse into just one linear layer - ReLU is what
  lets the network learn genuinely non-linear patterns.
- `nn.Dropout(0.1)` - same safeguard against over-memorizing as before.
- `nn.Linear(128, 3)` - the final layer, producing 3 numbers, one per class.

## Putting it together: the forward pass

"Forward pass" means: here's how data flows through the model, start to
finish.

```python
def forward(self, x):
    # x shape: (Batch, 2, 10)   <- a batch of jet pairs, 10 features each
    x = self.embedding(x)        # -> (Batch, 2, 64)  translate each jet
    x = self.transformer(x)      # -> (Batch, 2, 64)  jets exchange info
    x_pooled = x.mean(dim=1)     # -> (Batch, 64)     average the 2 jets
    out = self.mlp(x_pooled)     # -> (Batch, 3)      final class scores
    return out
```

Reading the shape comments is the easiest way to follow along: we start
with 2 jets described by 10 raw numbers each, expand each jet's
description to 64 richer numbers, let the two jets exchange information
twice, average them into one 64-number event summary, and boil that down
to 3 scores - one per possible answer.

::::::::::::::::::::::::::::::::::::: challenge

## Question

Q: A batch contains 32 jet pairs (`Batch = 32`). What is the shape of `x` right after `self.embedding(x)`, right after `self.transformer(x)`, and right after `x.mean(dim=1)`?

:::::::::::::::: solution

A: After `self.embedding(x)`: `(32, 2, 64)` - embedding expands each jet's 10 numbers to 64 but doesn't change batch size or jets per event. After `self.transformer(x)`: still `(32, 2, 64)` - self-attention exchanges information but doesn't change shape, only values. After `x.mean(dim=1)`: `(32, 64)` - averaging over the "2 jets" dimension collapses it away, leaving one 64-number summary per event.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::

## Quick recap
- The embedding layer translates each jet's 10 raw numbers into a richer 64-number internal description.
- Self-attention lets the two jets exchange information about each other - using 4 parallel "attention heads," repeated over 2 stacked layers.
- Mean pooling merges the two jets' descriptions into one summary per event.
- A small MLP turns that summary into 3 final class scores.
- Next: [Training the Model](07-training-the-model.md) - how the model actually learns from examples.

:::::: keypoints
- The embedding layer translates each jet's 10 raw numbers into a richer 64-number internal description.
- Self-attention lets the two jets exchange information about each other, using 4 parallel attention heads, repeated over 2 stacked layers.
- Mean pooling merges the two jets' descriptions into one summary per event, and keeps the model's answer independent of which jet was listed first.
- A small MLP turns that summary into 3 final class scores.
::::::
