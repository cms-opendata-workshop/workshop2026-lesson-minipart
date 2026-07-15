---
title: "Preparing the Data"
teaching: 20
exercises: 10
---

:::::: questions
- How do the three separate datasets (ttHTobb, ttHTocc, QCD) get combined into one dataset the model can train on?
- Why do we hold back part of the data as a test set instead of training on everything?
- Why does every feature need to be put on the same numerical scale before training?
- Why is it important to fit the scaler only on training data, never on test data?
- Why does the model see data in small shuffled batches instead of all at once?
::::::

:::::: objectives
- Combine the three extracted datasets into single `X` and `y` arrays.
- Split data into training and test sets while preserving class proportions.
- Explain why feature scaling matters for a neural network, and apply `StandardScaler` correctly.
- Explain the difference between `fit_transform` and `transform`, and why the distinction prevents data leakage.
- Convert NumPy arrays into PyTorch tensors and batches using `TensorDataset` and `DataLoader`.
::::::

## Run this first

Run this cell first, then read on to see what each part does.

```python
# Extract features (adjust max_events to None when ready for full training)
X_bb, y_bb = extract_features(TTHTOBB_PATH, label=0, is_signal=True, max_events=50000)
X_cc, y_cc = extract_features(TTHTOCC_PATH, label=1, is_signal=True, max_events=50000)
X_qcd, y_qcd = extract_features(QCD_BCTOE_PATH, label=2, is_signal=False, max_events=50000)

# Combine datasets
X = np.concatenate([X_bb, X_cc, X_qcd], axis=0)
y = np.concatenate([y_bb, y_cc, y_qcd], axis=0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features (Transformers are sensitive to scale)
# We flatten to (N*2, Features) to fit the scaler, then reshape back
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, len(FEATURE_NAMES))
X_test_flat = X_test.reshape(-1, len(FEATURE_NAMES))

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, 2, len(FEATURE_NAMES))
X_test_scaled = scaler.transform(X_test_flat).reshape(-1, 2, len(FEATURE_NAMES))

# Convert to PyTorch tensors
train_data = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
```

```output
Loaded label 0: 36025 events
Loaded label 1: 37295 events
Loaded label 2: 50000 events
```

`TTHTOBB_PATH`, `TTHTOCC_PATH`, and `QCD_BCTOE_PATH` are the three
streaming URLs saved in [Working in Google Colab](02-colab-and-data-access.md).
`extract_features()` is the function you already ran in
[Finding the Truth Labels](04-finding-the-truth-labels.md); it reads its
`filepath` argument the same way whether it's a local path or a `root://`
streaming URL. The exact counts above will vary slightly depending on the
files read, but stay in the same ballpark as the roughly 36,000 Hbb /
37,000 Hcc / 50,000 QCD split reflected in the test-set totals used
throughout [Evaluating the Model](08-evaluating-the-model.md).

Once `extract_features()` has turned all three files into arrays of jet
pairs and labels, three things still need to happen before we can hand
this to a neural network - all in the code you just ran above.

## Step 1 - Combine everything into one big pile

The previous episode's `extract_features()` returns a separate `X` and
`y` array for each dataset. Before anything else, those three pairs need
to become one combined dataset:

```python
X = np.concatenate([X_bb, X_cc, X_qcd], axis=0)
y = np.concatenate([y_bb, y_cc, y_qcd], axis=0)
```

`X` now holds every jet pair from all three datasets, shape
`(total_events, 2, 10)` - however many events, 2 jets each, 10 numbers
per jet. `y` holds the matching label (0, 1, or 2) for each one.

## Step 2 - Split into training data and test data

With one combined dataset, we set aside some of it and never show it to
the model during training, so there's a fair way to check afterward
whether it actually learned something useful:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

We hold back 20% of the data and never train on it. A model that just
*memorized* its training examples would score perfectly on those examples
without learning anything useful - testing on data it's never seen is the
only honest way to check.

- `random_state=42` makes the split repeatable - anyone running this code
  gets the same split.
- `stratify=y` keeps the same *proportion* of Hbb, Hcc, and QCD examples
  in both the training and test sets.

::: callout
`stratify=y` keeps proportions equal, but doesn't force equal counts to
begin with. In this dataset, QCD selection keeps a higher fraction of
events than the Hbb/Hcc truth-matching step, so the final dataset has
somewhat more QCD examples than Hbb or Hcc. You'll see this reflected in
the evaluation episode's confusion matrix - more true-QCD test events
than true-Hbb or true-Hcc - and it comes from here, not a training bug.
:::

## Step 3 - Put every feature on the same "ruler" (scaling)

`Jet_pt` might be a number like `85.0` (GeV), while an energy fraction
like `Jet_chHEF` is always between `0.0` and `1.0`. Fed raw into a neural
network, it would initially treat the *size* of a number as more
important just because its values are bigger, not because it's actually
more useful.

The fix is **standardization**: shift and rescale every feature so its
average is `0` and its spread (standard deviation) is `1` across the
training set - the same "how many steps away from average" scale for
every feature.

```python
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, len(FEATURE_NAMES))     # (n_events*2, 10)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, 2, len(FEATURE_NAMES))
X_test_scaled = scaler.transform(X_test_flat).reshape(-1, 2, len(FEATURE_NAMES))
```

Two important details:

- We `.reshape(-1, 10)` first because `StandardScaler` expects a plain
  table of rows and columns, not a 3D block, so we temporarily flatten
  "2 jets" into extra rows, scale, then reshape back.
- We call `.fit_transform()` on the **training** data (learn the average
  and spread from it, then apply it), but only `.transform()` on the
  **test** data. The test set must be scaled using the training set's
  numbers, not its own - otherwise we'd leak a peek at the test data into
  how we prepared the training data, which makes results look better than
  they really are.

## Step 4 - Turn everything into PyTorch tensors, in batches

PyTorch doesn't work directly on NumPy arrays - it uses its own array
type, the **tensor**, which supports the bookkeeping needed for training
(more in [Training the Model](07-training-the-model.md)).

```python
train_data = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
```

Instead of showing the model all training examples at once (slow) or one
at a time (unstable), we show it small **batches** of 256 examples.
`DataLoader` chops the dataset into batches and, with `shuffle=True`,
mixes up the order every epoch so the model doesn't learn something from
the order examples happen to be stored in.

::::::::::::::::::::::::::::::::::::: challenge

## Question

Q: Suppose `X_test_scaled = scaler.transform(X_test_flat)...` was accidentally changed to `X_test_scaled = scaler.fit_transform(X_test_flat)...` instead. Would this cause an error? What would actually go wrong?

:::::::::::::::: solution

A: No error - `fit_transform` is valid on its own, so the code would run. The problem is subtler: `fit_transform` would compute a new average and spread *from the test set itself* instead of using the training set's numbers. The test set would no longer be scaled the way the model was trained to expect, and information about the test set's own distribution would leak into how it's prepared. Results would look different, usually better than they honestly should, with no error message pointing to why.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::

## Quick recap
- Combine all three datasets, then split 80/20 into train/test, keeping the class proportions equal on both sides (`stratify`).
- Scale every feature to the same "average 0, spread 1" footing, fitting the scaler only on training data.
- Convert to PyTorch tensors and feed the model small shuffled batches at a time, not the whole dataset at once.
- Next: [Building MiniParT](06-building-mini-part.md) - building the model itself.

:::::: keypoints
- Combine all three datasets, then split 80/20 into train/test, keeping the class proportions equal on both sides with `stratify`.
- Scale every feature to the same average-0, spread-1 footing, fitting the scaler only on training data.
- Convert to PyTorch tensors and feed the model small shuffled batches at a time, not the whole dataset at once.
- QCD examples slightly outnumber Hbb and Hcc examples after selection, because truth-matching discards more signal events than the QCD selection discards background events.
::::::
