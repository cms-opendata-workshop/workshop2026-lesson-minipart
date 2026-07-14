---
title: 'Instructor Notes'
---

## Timing

Across all 9 episodes, this lesson totals approximately 250 minutes
(about 4 hours) of combined teaching and exercise time, based on the
`teaching` and `exercises` values set in each episode's front matter. This
comfortably fills a full workshop day, or can be split across two
half-day sessions with a natural break after
[Preparing the Data](../episodes/05-preparing-the-data.md), before
learners start building the model itself.

[Building MiniParT](../episodes/06-building-mini-part.md) is the longest
and most conceptually dense episode (45 minutes). Consider giving it more
buffer time than its listed duration suggests, especially for learners
seeing self-attention for the first time.

## Likely points of confusion

**Why Hbb vs. Hcc is harder than signal vs. background.** This is the
single most important physics idea in the lesson, introduced in
[The Big Picture](../episodes/01-introduction.md) and confirmed with real
numbers in [Evaluating the Model](../episodes/08-evaluating-the-model.md).
Learners sometimes expect the model to struggle most at telling a Higgs
event apart from background, since that sounds like the "harder" question
at a glance. The actual result (QCD separates cleanly, Hbb and Hcc are
confused with each other far more often) surprises people until they
recall that bottom and charm quarks are physically similar to each other,
while both are quite different from a generic QCD jet. It may help to
pause on the confusion matrix in the evaluation episode and ask learners
to predict, before revealing the real numbers, which pair of classes they
expect the model to confuse most.

**Streaming paths defined once, reused everywhere.** `TTHTOBB_PATH`,
`TTHTOCC_PATH`, and `QCD_BCTOE_PATH` are defined once in
[Working in Google Colab](../episodes/02-colab-and-data-access.md) and
used directly, unchanged, in every later episode's code - there is no
local-path substitution step. The one real gotcha is Colab session
state: these variables only exist in memory for the current runtime. If
a learner's session disconnects and they reconnect, or they skip ahead
without running episode 2's cells first, later code that references
`TTHTOBB_PATH` will fail with a `NameError`, not a file-not-found error.
If that happens, the fix is simply to re-run the path-defining cell from
episode 2, not to look for a missing file.

**Colab session timeouts during training.** Training MiniParT for 10
epochs is fast (minutes, not hours) on the datasets used in this lesson,
but Colab sessions can still disconnect after periods of inactivity, or
if a learner leaves a cell running and switches away for a long time.
Warn learners not to close their browser tab or let the machine go to
sleep mid-training, and mention that if a session does disconnect, the
notebook needs to be re-run from the top, including the `!pip install`
cell and the data streaming cells, since a fresh Colab session starts
with nothing installed and nothing loaded into memory.

**ΔR and the eta-phi coordinate system.** The geometric matching in
[Finding the Truth Labels](../episodes/04-finding-the-truth-labels.md) is
often the first time learners work with `eta` and `phi` as a coordinate
system rather than abstract numbers. The "latitude and longitude"
comparison introduced in the previous episode is worth repeating here,
along with the `delta_phi` wraparound issue, since it is easy to
underestimate how often phi values land near the 0-to-2π boundary in a
real dataset.

**Data leakage in scaling.** The distinction between `fit_transform`
(training data) and `transform` (test data) in
[Preparing the Data](../episodes/05-preparing-the-data.md) is a subtle
point that is easy to nod along to without really absorbing. The
challenge exercise in that episode is designed to surface this directly;
if time is short, it is worth keeping that exercise rather than cutting
it, since this mistake is common in learners' own future projects.

**Why DeepJet/DeepCSV variables are excluded.** Learners with some prior
exposure to CMS analyses sometimes ask why the lesson does not just use
existing, more accurate jet-tagging scores as input features. The
explanation in [What Is a Jet?](../episodes/03-what-is-a-jet.md) addresses
this directly: using another tagger's output as a feature would let
MiniParT copy an existing answer instead of learning to separate the
three classes from more basic information, which would defeat the
purpose of this lesson as a from-scratch teaching example.

## Notre Dame CMS Open Data Workshop framing

This lesson is designed to be taught in the same style as the other CMS
Open Data Workshop lessons: concept first, then code, with the physics
motivation kept front and center rather than treated as a footnote to the
machine learning content. When teaching this lesson alongside other CMS
Open Data Workshop material, it works well as a capstone: it draws on the
same NanoAOD data format and CERN Open Data access patterns used
elsewhere in the workshop, applied to a machine learning classification
task instead of a traditional physics analysis.
