"""Train and evaluate a demographic-parity-constrained classifier.

Run from the project root with:

    python -m examples.quickstart
"""

import numpy as np
import pandas as pd
import torch

from fairget import DemographicParity, FairClassifier


def make_proxy_dataset(n=3000, seed=11):
    """Create data in which a usable feature is correlated with group membership.

    The sensitive column is excluded from the model inputs, but ``proxy_score``
    lets an unconstrained classifier learn different selection rates. This makes
    the example suitable for checking whether the constraint changes the model.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n)
    skill = rng.normal(size=n)
    proxy_score = group + rng.normal(scale=0.30, size=n)
    latent_score = (
        2.5 * skill
        + 3.0 * group
        - 1.5
        + rng.normal(scale=0.60, size=n)
    )
    outcome = (latent_score > 0).astype(int)

    return pd.DataFrame({
        "skill": skill,
        "proxy_score": proxy_score,
        "group": np.where(group == 1, "B", "A"),
        "outcome": np.where(outcome == 1, "yes", "no"),
    })


def split_rows(df, train_fraction=0.70, seed=19):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    split = int(train_fraction * len(df))
    train = df.iloc[order[:split]].reset_index(drop=True)
    test = df.iloc[order[split:]].reset_index(drop=True)
    return train, test


def make_logistic_model(input_size):
    """Return a fresh probability-producing model for each training run."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 1),
        torch.nn.Sigmoid(),
    )


def main():
    train_df, test_df = split_rows(make_proxy_dataset())

    classifier = FairClassifier(
        constraints=[DemographicParity(delta=0.8)],
        model_factory=make_logistic_model,
        alpha=5.0,
        epochs=500,
        satisfied_tol=0.0,
        seed=1,
    )
    classifier.fit(
        train_df,
        target="outcome",
        sensitive="group",
        positive_label="yes",
        privileged_group="B",
    )

    print(classifier.fairness_report())
    print()
    print(classifier.evaluate(test_df, split_name="test"))

    new_rows = test_df.sample(5, random_state=1).drop(columns=["outcome"])
    print("\nPredictions on five held-out rows:", classifier.predict(new_rows))


if __name__ == "__main__":
    main()
