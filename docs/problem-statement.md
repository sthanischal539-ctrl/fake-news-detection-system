# Problem Statement

## The problem
The volume of online news and social content has made it trivial to publish and
spread fabricated or misleading articles at scale. Fabricated news stories are
written to look like legitimate reporting, which makes them hard to distinguish
from real journalism by eye — especially when a reader only skims the headline.

This project builds a **Fake News Detection System** that analyses the **full
article text** (not just the headline) and classifies it as **Fake** or **Real**,
while explaining *why* the model reached its decision.

## Why it matters
- **Scale:** No human team can manually fact-check the full firehose of online news.
- **Speed:** Misinformation spreads fastest in the first hours after publication,
  before fact-checkers respond. An automated signal can flag content for review.
- **Transparency:** A black-box "Fake/Real" verdict is not trustworthy on its own.
  Showing which words/features drove a prediction lets a human judge whether the
  model is picking up on meaningful signal or on spurious artefacts (e.g. a
  dataset's formatting quirks).
- **Education:** Surfacing the reasoning helps users build intuition about the
  linguistic patterns that often accompany unreliable content.

## Scope

### In scope
- English-language news articles supplied as **full body text**.
- Binary classification: **Fake** vs **Real**.
- Three model families for comparison:
  - Logistic Regression on TF-IDF features
  - Linear SVM on TF-IDF features
  - A transformer baseline (e.g. DistilBERT)
- Proper held-out evaluation: Accuracy, Precision, Recall, F1, confusion matrix.
- Per-prediction explainability (top influential words/features).
- A Streamlit web interface and an evaluation dashboard.

### Out of scope (for this prototype)
- Non-English or multilingual articles.
- Real-time claim verification against external fact-check databases or the live web.
- Image, video, or audio (deepfake) analysis.
- Source-reputation / network-propagation signals (who shared it, from where).
- Multi-class labels (satire, propaganda, clickbait, bias). Binary only.
- Production hardening, authentication, scaling, and rate limiting.

## Critical disclaimer — model output is a signal, not proof

> **A prediction from this system does NOT confirm whether a news article is
> actually fake or real.** It is a **probability-based signal** produced by a
> statistical model trained on a specific dataset. It can be confidently wrong.

Key limitations the user must understand:

- The model only learns patterns present in its **training data**. Articles about
  new topics, in new styles, or from new sources may be misclassified.
- Models can latch onto **dataset artefacts** (publication names, date formats,
  boilerplate) rather than genuine indicators of truthfulness. Explainability is
  included precisely so these artefacts become visible.
- A high confidence score reflects the model's **certainty about its learned
  patterns**, not the actual truth of the claim.
- Truth is determined by **verifiable facts and trusted reporting**, not by writing
  style. This tool measures style and vocabulary, not factual accuracy.

**Every prediction in the web interface will be accompanied by a visible disclaimer
instructing the user to verify any claim against trusted, authoritative sources
before acting on it.** The system is a triage/assistance aid, not an arbiter of truth.
