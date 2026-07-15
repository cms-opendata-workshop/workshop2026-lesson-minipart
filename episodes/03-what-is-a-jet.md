---
title: "What Is a Jet?"
teaching: 20
exercises: 10
---

:::::: questions
- What is a jet, and why does the CMS detector record jets instead of individual quarks?
- What are the 10 numbers used to describe each jet, and what physical property does each one capture?
- Why does MiniParT use pre-computed summary numbers per jet instead of raw particle-level data?
- Why does this lesson deliberately avoid using DeepJet/DeepCSV tagger scores as features?
- Where does this jet data actually come from, and how do we read it in Python?
::::::

:::::: objectives
- Describe what a jet is and why it is what the detector actually measures.
- List the 10 features used to describe each jet and group them by what physical property they describe.
- Explain why MiniParT uses summary numbers instead of raw particle-level detail.
- Explain why the lesson avoids DeepJet/DeepCSV variables as input features.
- Read jet data out of a NanoAOD file using `uproot`.
::::::

## Run this first

```python
import uproot
import awkward as ak
import vector
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

vector.register_awkward()

# We avoid DeepJet/DeepCSV variables as requested.
# Using kinematics + energy fractions + pileup/multiplicity info.
FEATURE_NAMES = [
    'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass',
    'Jet_chHEF', 'Jet_neHEF', 'Jet_chEmEF', 'Jet_neEmEF',
    'Jet_nConstituents', 'Jet_puId'
]

# Labels: 0 = Hbb, 1 = Hcc, 2 = QCD
```

---
*Run the block above first, then read on to see what each part does.*

This block gathers the imports used across the rest of this lesson,
including `vector.register_awkward()`, which is needed before the
four-vector matching used in the next episode will work, and defines
`FEATURE_NAMES`, the exact list of 10 jet numbers explained below.

DeepJet and DeepCSV are CMS's own existing, more sophisticated jet-tagging
algorithms - already trained to estimate things like "how likely is this
jet to contain a bottom quark." Feeding their output into MiniParT would
let the model just copy an existing tagger's answer instead of learning
to separate Hbb, Hcc, and QCD from more basic information itself, which
would defeat the point of building a classifier from scratch.

## The detector, in one picture (with words)

When a collision happens inside CMS, quarks and other particles shoot
outward in all directions. A single quark can't exist alone for long - as
it flies outward, it drags a cloud of other particles with it, the way a
fast-moving truck kicks up a spray of gravel behind it. **That whole
spray, moving roughly together, is called a jet.** For the full
detector-level picture - the clustering algorithm, pileup removal, and
the complete NanoAOD `Jet_*` branch list - see the CMS Open Data
Workshop's
[jets and MET lesson](https://cms-opendata-workshop.github.io/workshop2026-lesson-physics-objects/instructor/04-jetmet.html);
this episode picks up from there with the 10 numbers MiniParT actually uses.

The detector doesn't record "a bottom quark went this way." It records
where dozens of individual particles landed and how much energy each one
carried. Reconstruction software then groups those particles back
together into a jet and computes useful summary numbers about it.
**Those summary numbers are what our model actually sees** - not the raw
particle hits.

## The 10 numbers we give the model

For every jet, we use these 10 features. You already defined this exact
list as `FEATURE_NAMES` in the code above, and the `extract_features()`
function built in the [next episode](04-finding-the-truth-labels.md) reuses
it:

### Where the jet is and how big it is

- **`Jet_pt`** (transverse momentum) - how hard the jet is moving,
  sideways relative to the beam pipe, in GeV. Bigger number = more
  energetic jet.
- **`Jet_eta`** (pseudorapidity) - the jet's angle relative to the beam
  pipe. `eta = 0` means straight out sideways; large values mean close to
  the beam line.
- **`Jet_phi`** - the jet's angle *around* the beam pipe, like a compass
  direction. `eta` and `phi` together pin down exactly which direction
  the jet flew, the way latitude and longitude pin down a spot on Earth.
- **`Jet_mass`** - the combined mass of everything in the jet. A tighter,
  simpler jet tends to have lower mass than one that's really two things
  overlapping.

### What the jet is made of

Every particle in the spray is either **charged** or **neutral**, and is
either a **hadron** (built from quarks) or behaves like **electromagnetic**
radiation (electrons and photons). These four "energy fractions" say what
fraction of the jet's total energy falls into each bucket, and always add
up to roughly 1:

- **`Jet_chHEF`** - Charged Hadron Energy Fraction.
- **`Jet_neHEF`** - Neutral Hadron Energy Fraction.
- **`Jet_chEmEF`** - Charged Electromagnetic Energy Fraction (mostly electrons/positrons).
- **`Jet_neEmEF`** - Neutral Electromagnetic Energy Fraction (mostly photons).

Different quark types tend to "hadronize" (turn into a jet) in slightly
different ways, so jets from heavier quarks are statistically a little
more likely to contain certain particles than background jets. No single
fraction gives it away, but the combination is a real clue.

### How the jet is put together

- **`Jet_nConstituents`** - how many individual particles were found
  inside the jet. A jet built from more sub-pieces "looks" different to
  the model than a tight jet built from just a few.
- **`Jet_puId`** - Pileup ID. Several unrelated proton collisions happen
  at almost the same instant inside the detector ("pileup"). This score
  estimates how likely the jet is genuine, versus stray junk from one of
  those other, unrelated collisions. Higher is usually more "real."

## Why not just look at raw particles?

The real, full-size Particle Transformer does exactly that - it looks at
every individual particle inside a jet, not just 10 summary numbers.
That's more powerful but much bigger and slower to train. MiniParT uses
10 pre-computed summary numbers instead, which is why it can train on a
laptop instead of a GPU cluster - the tradeoff is some detail gets
thrown away.

## Where this data lives

These numbers come straight out of **NanoAOD** files - CMS's compact,
public data format, stored as [ROOT](https://root.cern.ch/) files. We
read them with `uproot`, which pulls out a "branch" (a column of data) by
name without needing any CERN-specific software installed, using the
`FEATURE_NAMES` list already defined in the code above:

```python
max_events = 50000

tree = uproot.open(TTHTOBB_PATH)["Events"]
events = tree.arrays(FEATURE_NAMES, entry_stop=max_events)
print("Number of events loaded:", len(events))
print("Fields loaded:", events.fields)
```

```output
Number of events loaded: 50000
Fields loaded: ['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_chHEF', 'Jet_neHEF', 'Jet_chEmEF', 'Jet_neEmEF', 'Jet_nConstituents', 'Jet_puId']
```

`TTHTOBB_PATH` is the streaming URL you saved in
[Working in Google Colab](02-colab-and-data-access.md); `uproot.open()`
reads directly from CERN over that URL, just like a local file path.
`tree.arrays(...)` reads the requested columns for every event into
memory, and `entry_stop=max_events` caps how many events to read, so you
can test on a small slice before running on everything - here it caps the
174,000 events in the file down to the 50,000 shown above. `events.fields`
confirms the 10 columns actually loaded match `FEATURE_NAMES`.

## Quick recap
- A jet is a spray of particles created when a quark flies out of a collision.
- We describe each jet with 10 numbers: 4 about its size/direction (`pt`, `eta`, `phi`, `mass`), 4 about what it's made of (the energy fractions), and 2 about its structure (`nConstituents`, `puId`).
- These numbers are read from public CMS NanoAOD files using `uproot`.
- Next: [Finding the Truth Labels](04-finding-the-truth-labels.md) - how do we know the *right answer* for each jet during training?

::::::::::::::::::::::::::::::::::::: challenge

## Question

Q: Two jets have the same `Jet_pt`, `Jet_eta`, and `Jet_phi`, but one has `Jet_nConstituents = 4` and the other has `Jet_nConstituents = 35`. What does that suggest about how the two jets are physically different?

:::::::::::::::: solution

A: `Jet_nConstituents` counts how many particles were reconstructed inside the jet. Four constituents is a tight, simple spray; 35 is a broader, more complex one. Two jets can carry the same overall momentum and direction while being made of very different numbers of particles - exactly the kind of structural difference `Jet_nConstituents` and the energy fractions are meant to capture, since `Jet_pt`, `Jet_eta`, and `Jet_phi` alone say nothing about internal structure.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::

:::::: keypoints
- A jet is a spray of particles created when a quark flies out of a collision.
- We describe each jet with 10 numbers: 4 about its size/direction (`pt`, `eta`, `phi`, `mass`), 4 about what it's made of (the energy fractions), and 2 about its structure (`nConstituents`, `puId`).
- These numbers are read from public CMS NanoAOD files using `uproot`.
- DeepJet/DeepCSV tagger scores are deliberately excluded as features, so the model learns to separate Hbb/Hcc/QCD from basic information instead of copying an existing tagger's answer.
::::::
