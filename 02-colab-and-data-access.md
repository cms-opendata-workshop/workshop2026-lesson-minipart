---
title: "Working in Google Colab"
teaching: 10
exercises: 0
---

:::::: questions
- Which packages does this lesson need installed in Colab, and how do you install them?
- How do you read a CMS Open Data file directly from CERN without downloading it?
- Which three files does this lesson use, and how do you check a stream actually worked?
::::::

:::::: objectives
- Install the packages this lesson needs inside a Google Colab notebook.
- Open a CMS Open Data file directly from CERN using `uproot`, without downloading it first.
- Identify the three files this lesson uses and confirm a stream opened correctly.
::::::

## Installing the packages this lesson needs

Colab comes with many common data science packages already installed,
including `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and
`torch`. It does not come with `uproot` (for reading ROOT files),
`fsspec-xrootd` (for streaming those files over the network - without it,
opening a `root://` URL below fails immediately), `awkward` (for handling
data where different events can have different numbers of particles), and
`vector` (for particle four-vectors). Run this in a Colab cell before
anything else in this lesson:

```python
!pip install uproot fsspec-xrootd awkward vector numpy torch scikit-learn matplotlib seaborn pandas
```

## Reading files directly from CERN

CMS Open Data files live on CERN's servers and can be streamed straight
into `uproot` over a network protocol called xrootd, instead of being
downloaded first. Hand `uproot` a `root://` URL instead of a local file
path, and it reads only the parts of the file it actually needs:

```python
import uproot

tthtobb_path = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2/260000/410F948C-6956-2D45-A170-DE6431E02281.root"

tree = uproot.open(tthtobb_path)["Events"]
print("Number of events:", tree.num_entries)
```

Nothing here gets downloaded to disk - `uproot` streams just the data it
needs, which is why this works comfortably inside Colab's storage limits.

## The three files this lesson uses

This lesson uses three CMS Open Data files, one each from the ttHTobb,
ttHTocc, and QCD_bcToE records, all confirmed to contain well over
100,000 events:

| Dataset | CERN Open Data record | File used in this lesson | Confirmed events |
|---|---|---|---|
| ttHTobb (Hbb signal) | [record 67645](https://opendata.cern.ch/record/67645) | `410F948C-6956-2D45-A170-DE6431E02281.root` | 174,000 |
| ttHTocc (Hcc signal) | [record 67651](https://opendata.cern.ch/record/67651) | `5C12D3AA-9311-B840-BB5D-4155D7FF66E4.root` | 137,000 |
| QCD_bcToE (background) | [record 63242](https://opendata.cern.ch/record/63242) | `A133135A-C83E-D245-846F-210C7AD2D29C.root` | 345,045 |

Save each file's full streaming path as a variable now, so you can reuse
it in later episodes:

```python
TTHTOBB_PATH = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2/260000/410F948C-6956-2D45-A170-DE6431E02281.root"

TTHTOCC_PATH = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/ttHTocc_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/50000/5C12D3AA-9311-B840-BB5D-4155D7FF66E4.root"

QCD_BCTOE_PATH = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2/270000/A133135A-C83E-D245-846F-210C7AD2D29C.root"
```

## Verifying it works

Before moving on, confirm a stream actually opens and contains the data
this lesson expects:

```python
tree = uproot.open(TTHTOBB_PATH)["Events"]
print("Number of events:", tree.num_entries)
```

This should print `174000`. If it prints a much smaller number, you have
the wrong file - check it against the table above before using it.

:::::: keypoints
- Colab does not preinstall `uproot`, `fsspec-xrootd`, `awkward`, or `vector`; install them with `!pip install` before running anything else in this lesson.
- `uproot.open()` on a `root://` URL streams a CMS file directly from CERN's servers, without downloading it.
- This lesson uses three specific files, one each from the ttHTobb, ttHTocc, and QCD_bcToE CERN Open Data records.
- Always check `tree.num_entries` after opening a file - a much smaller count than expected means it's the wrong file.
::::::
