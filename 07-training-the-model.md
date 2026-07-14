---
title: "Training the Model"
teaching: 20
exercises: 10
---

:::::: questions
- What do the loss function and optimizer actually do during training?
- What happens, step by step, inside the training loop for one batch of data?
- What is an epoch, and why do we repeat the training loop for several of them?
- Why is training accuracy not enough to trust on its own?
::::::

:::::: objectives
- Explain the roles of `CrossEntropyLoss` and `AdamW` in training MiniParT.
- Walk through the six steps that happen for every batch during training.
- Define "epoch" and explain why the model trains over several of them.
- Explain why training accuracy alone cannot tell you whether the model generalizes.
::::::

The model from [Building MiniParT](06-building-mini-part.md) starts out
knowing nothing - its weights are randomly initialized. Training is the
repeated process of showing it examples, checking how wrong its guesses
are, and nudging its weights to be a little less wrong next time. This is
what the code built in this episode below does.

## Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

- **`device`** - trains on a GPU (`cuda`) if available, since GPUs do this
  math much faster than a CPU; otherwise falls back to CPU. Either works
  for a model this small, just at different speeds.
- **`criterion` (the loss function)** - `CrossEntropyLoss` measures "how
  wrong was the guess," comparing the model's 3 class scores against the
  true label. Low if confidently correct, high if confidently wrong -
  think of it as an automatic grader.
- **`optimizer`** - `AdamW` adjusts the model's weights based on the loss,
  like an editor deciding how to nudge every tunable number after seeing
  the grade.
  - **`lr=1e-3`** (learning rate) - how big a nudge to make each step.
    Too big and the model overshoots; too small and it learns painfully
    slowly. `0.001` is a common starting point.
  - **`weight_decay=1e-4`** - mild extra pressure discouraging weights
    from growing unnecessarily large, another safeguard against
    over-memorizing.

## The training loop

This is the core loop that updates the model's weights, once per batch,
every epoch:

```python
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
```

One **epoch** is one full pass through every batch in the training set;
we do 10 (`epochs = 10`). For every batch of 256 jet pairs:

1. **`model.train()`** - tells the model we're training, switching on
   dropout (see [Building MiniParT](06-building-mini-part.md)).
2. **`optimizer.zero_grad()`** - clears leftover gradient information from
   the previous batch, so nudges don't incorrectly pile on top of each other.
3. **`outputs = model(batch_x)`** - the forward pass: run this batch
   through the model and get 3 class scores for each example.
4. **`loss = criterion(outputs, batch_y)`** - compares those scores to the
   true labels and produces one number: how wrong, on average.
5. **`loss.backward()`** - **backpropagation**: PyTorch works backward
   through every layer and calculates how much each weight contributed to
   the error, so we know which direction to nudge each one.
6. **`optimizer.step()`** - applies those nudges, updating every weight a
   small amount in the direction that should reduce the loss.

Repeat for every batch, for 10 epochs, and the model gradually gets
better at telling Hbb, Hcc, and QCD jet pairs apart.

::: callout
`CrossEntropyLoss` treats every example equally by default. As the
previous episode noted, this dataset ends up with slightly more QCD
examples than Hbb or Hcc after selection, so the model sees somewhat more
QCD during training. The imbalance here is small enough not to need
special handling, but it's worth checking in any classification problem
with unequal class sizes - left unchecked in a more imbalanced dataset, a
model can learn to just favor the larger class.
:::

## Watching it learn

```python
train_acc = 100. * correct / total
print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
```

After each epoch, we print the average loss and training accuracy (the
percentage of training examples the model got right during that epoch).
You should generally see loss go down and accuracy go up epoch over
epoch - the model visibly learning.

This is *training* accuracy, measured on data the model has already
seen - useful to sanity-check that learning is happening, but not a fair
report card. For that, we need data the model has never seen, which the
next episode covers.

::::::::::::::::::::::::::::::::::::: challenge

## Question

Q: Suppose `optimizer.zero_grad()` was accidentally left out of the training loop. Would the code raise an error? What would you expect to happen to training over the 10 epochs instead?

:::::::::::::::: solution

A: No error would be raised. Without `optimizer.zero_grad()`, the gradient information from `loss.backward()` would accumulate on top of the previous batch's instead of starting fresh each time. The weight updates in `optimizer.step()` would then be based on this incorrectly piled-up information, so training would likely become unstable: loss might behave erratically instead of steadily decreasing, and accuracy might fail to improve smoothly across epochs.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::

## Quick recap
- `CrossEntropyLoss` grades how wrong the model's guesses are; `AdamW` decides how to adjust the model's weights in response.
- Each batch: clear old gradients → forward pass → compute loss → backpropagate → update weights.
- One full pass through all batches is an epoch; we repeat for several epochs so the model keeps improving.
- Training accuracy is a useful sanity check, but not a fair test - see [Evaluating the Model](08-evaluating-the-model.md) for that.

## Full code for this lesson

Copy this into your own Jupyter notebook cell(s), in order, as you go.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
```

:::::: keypoints
- `CrossEntropyLoss` grades how wrong the model's guesses are; `AdamW` decides how to adjust the model's weights in response.
- Each batch follows the same six steps: clear old gradients, forward pass, compute loss, backpropagate, update weights.
- One full pass through all batches is an epoch; we repeat for several epochs so the model keeps improving.
- Training accuracy is a useful sanity check, but not a fair test, since it is measured on data the model has already seen.
::::::
