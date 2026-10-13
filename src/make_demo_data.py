"""Generate a small SYNTHETIC demo dataset so the pipeline runs out-of-the-box.

This is NOT real news. It exists only so you can train, evaluate, and launch the
app without first downloading a large dataset. The two classes are written in
deliberately different styles (sensational/clickbait vs. measured/attributed
reporting) so the classical models learn a genuine — if easy — signal.

For real results, download the ISOT dataset (see docs/dataset.md) and place
True.csv / Fake.csv in data/raw/; data_loader.py will use it automatically.
"""
from __future__ import annotations

import random

import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Vocabulary / templates
# --------------------------------------------------------------------------- #
TOPICS = [
    "the economy", "the new vaccine", "the upcoming election", "the tech company",
    "the climate summit", "the local hospital", "the stock market", "the space mission",
    "the water supply", "the education reform", "the national budget", "the trade deal",
    "the football club", "the research study", "the city council", "the power grid",
]

REAL_OPENERS = [
    "According to officials, {topic} showed measured progress this quarter.",
    "Researchers reported that {topic} was reviewed under standard procedures.",
    "A spokesperson said {topic} would be monitored in the coming weeks.",
    "Data released on Tuesday indicated {topic} remained broadly stable.",
    "Analysts noted that {topic} met expectations set earlier this year.",
    "The committee confirmed {topic} was assessed by independent experts.",
]
REAL_BODY = [
    "The report, published after peer review, cited figures from the agency.",
    "Officials declined to speculate and said further analysis was ongoing.",
    "The findings were consistent with earlier estimates, the statement added.",
    "Several economists described the change as modest but within forecasts.",
    "A follow-up review is scheduled, according to two people familiar with the matter.",
    "The department said it would publish the full methodology next month.",
    "Local authorities confirmed the figures and urged residents to stay informed.",
    "The study followed established protocols and was funded by a public grant.",
]

FAKE_OPENERS = [
    "SHOCKING: {topic} is a SECRET plot they DON'T want you to know!!!",
    "You won't BELIEVE what insiders just exposed about {topic}!",
    "BREAKING miracle: {topic} CURE hidden by the elites for years!",
    "They are LYING to you about {topic} — the TRUTH is finally out!",
    "This one WEIRD trick about {topic} has doctors FURIOUS!",
    "URGENT: share before they DELETE this bombshell about {topic}!",
]
FAKE_BODY = [
    "Anonymous sources claim the government covered everything up overnight.",
    "Experts are SILENCED and the mainstream media refuses to report it!!!",
    "Thousands are waking up to this conspiracy that changes EVERYTHING.",
    "Big corporations paid millions to keep this secret buried forever.",
    "Click now before the powerful elites scrub it from the internet!!!",
    "A whistleblower leaked proof that nobody in power dares to deny.",
    "It is a guaranteed miracle cure that the establishment fears most.",
    "Wake up people — this shocking truth will blow your mind completely!",
]


def _make_article(rng: random.Random, fake: bool) -> str:
    topic = rng.choice(TOPICS)
    openers = FAKE_OPENERS if fake else REAL_OPENERS
    body = FAKE_BODY if fake else REAL_BODY
    sentences = [rng.choice(openers).format(topic=topic)]
    sentences += [rng.choice(body) for _ in range(rng.randint(3, 6))]
    rng.shuffle(sentences[1:])  # vary body order, keep opener first
    return " ".join(sentences)


def generate(n_per_class: int = 500, seed: int = config.SEED) -> pd.DataFrame:
    """Return a balanced DataFrame with columns: text, label (0=REAL, 1=FAKE)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": _make_article(rng, fake=False), "label": config.LABEL_REAL})
        rows.append({"text": _make_article(rng, fake=True), "label": config.LABEL_FAKE})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    out = config.RAW_DIR / "demo_news.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic demo rows to {out}")


if __name__ == "__main__":
    main()
