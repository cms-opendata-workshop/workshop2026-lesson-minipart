---
title: Setup
---

This lesson runs entirely in [Google Colab](https://colab.research.google.com/).
There is nothing to install on your own computer, and no dataset to
download before you begin. All CMS Open Data files this lesson uses are
streamed directly from CERN's servers while your Colab notebook is
running.

## What you need

- A Google account, to open and run notebooks in Google Colab.
- A web browser.
- Nothing else. No local Python installation, no virtual environment, and
  no downloaded data files are required.

## Opening a Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com/) and
sign in with your Google account, then choose "New notebook." Everything
in this lesson can be typed or pasted into cells in that notebook, in the
order the episodes present it.

## Installing the packages this lesson needs

Colab already has several common data science packages installed,
including `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and
`torch`. It does not have `uproot`, `fsspec-xrootd`, `awkward`, or
`vector`, which this lesson uses to read CMS data files. Run this in the
first cell of your Colab notebook, before anything else in this lesson:

```python
!pip install uproot fsspec-xrootd awkward vector numpy torch scikit-learn matplotlib seaborn pandas
```

Full details on why these specific packages are needed, and how the rest
of this lesson reads CMS data without downloading it, are covered in
[Working in Google Colab](../episodes/02-colab-and-data-access.md), the
second episode of this lesson. Start there once your notebook is open and
the install command above has finished running.

## No data download required

This lesson uses three CMS Open Data files (described in full in
[Working in Google Colab](../episodes/02-colab-and-data-access.md)). None
of them need to be downloaded. Instead, this lesson reads them directly
from CERN's `eospublic.cern.ch` server using `uproot`, over a network
protocol called xrootd, which streams only the parts of a file that are
actually needed rather than requiring the whole file to sit on disk. This
keeps the lesson well within Colab's storage limits and means the exact
same code works whether you are running this lesson today or next year,
without maintaining a local copy of any dataset.

::: callout
If you would rather work locally instead of in Colab, everything in this
lesson still works in a local Jupyter notebook or plain Python script.
Replace the `!pip install ...` command above with the same command
without the leading `!`, run in a terminal, and the same streaming file
paths from [Working in Google Colab](../episodes/02-colab-and-data-access.md)
will work identically outside of Colab.
:::
