---
site: sandpaper::sandpaper_site
---

This lesson builds, from scratch, a small transformer model that classifies
pairs of jets from CMS Open Data as coming from a Higgs boson decaying to
two bottom quarks (Hbb), a Higgs boson decaying to two charm quarks (Hcc),
or ordinary QCD background with no Higgs boson involved. This model,
MiniParT, is a scaled-down version of the Particle Transformer architecture
used in real CMS physics analyses: small enough to train in minutes on a
free Google Colab session, but built using the same ideas as the full-size
version.

This lesson grew out of the Notre Dame CMS Open Data Workshop and follows
the same format as the CMS Open Data Workshop lessons. It runs entirely in
Google Colab. You do not need to install Python locally or download any
data files: the [Working in Google Colab](episodes/02-colab-and-data-access.md)
episode covers everything you need, including streaming the CMS data files
directly from CERN's servers.

By the end of this lesson, you will have read real CMS Open Data files,
built the truth labels needed to train a classifier, built and trained a
small transformer model, and evaluated whether it actually learned to tell
Hbb, Hcc, and QCD jet pairs apart, including where it succeeds and where
the underlying physics makes the problem genuinely hard.

## Prerequisites

- Basic Python: variables, functions, loops, and reading simple code.
- Some familiarity with the general idea of machine learning (for example,
  that a model is trained on examples and then makes predictions on new
  data). No prior experience with neural networks or transformers is
  assumed; those concepts are introduced from scratch in this lesson.
- A Google account, used to open and run notebooks in Google Colab.
- No prior particle physics background is required. The physics concepts
  needed (jets, quarks, the Higgs boson, and how CMS records collisions)
  are introduced in the first two episodes.

::::::::::::::::::::::::::::::::::::::: prereq

## Setup

Before starting this lesson, see [Setup](learners/setup.md) for how to open
a Colab notebook and install the packages this lesson needs. There is
nothing to download or install on your own computer.

::::::::::::::::::::::::::::::::::::::::::::::::
