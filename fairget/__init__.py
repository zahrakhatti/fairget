"""Tools for training binary classifiers with fairness constraints.

Reports compare an unconstrained baseline with the constrained model and show
the requested and observed fairness values on thresholded predictions.

Basic usage
-----------
    from fairget import FairClassifier, DemographicParity

    clf = FairClassifier(constraints=[DemographicParity(delta=0.8)])
    clf.fit(
        train_df,
        target="hired",
        sensitive="group",
        positive_label="yes",
        privileged_group="B",
    )
    print(clf.evaluate(test_df, split_name="test"))
    preds = clf.predict(df_new)
"""

from fairget.criteria import (
    DemographicParity,
    DisparateImpact,
    EqualOpportunity,
)
from fairget.model import FairClassifier
from fairget.report import FairnessReport
from fairget._version import __version__

__all__ = [
    "FairClassifier",
    "FairnessReport",
    "DemographicParity",
    "DisparateImpact",
    "EqualOpportunity",
]
