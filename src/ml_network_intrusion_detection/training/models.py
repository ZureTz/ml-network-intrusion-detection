"""可复现的小规模候选模型集合。"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_network_intrusion_detection.data.schema import CATEGORICAL, NUMERIC


def candidates(profile: str) -> list[dict]:
    """返回顺序固定的候选配置；smoke 只运行每类一个轻量配置。"""
    result = [{"name": "dummy", "family": "dummy", "params": {}}]
    for c in [1] if profile == "smoke" else [0.1, 1, 10]:
        for weight in [None] if profile == "smoke" else [None, "balanced"]:
            result.append(
                {
                    "name": f"lr-c{c}-{weight}",
                    "family": "logistic_regression",
                    "params": {"C": c, "class_weight": weight},
                }
            )
    for depth in [12] if profile == "smoke" else [None, 20]:
        for leaf in [1] if profile == "smoke" else [1, 5]:
            result.append(
                {
                    "name": f"rf-depth{depth}-leaf{leaf}",
                    "family": "random_forest",
                    "params": {
                        "n_estimators": 20 if profile == "smoke" else 200,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                    },
                }
            )
    return result


def build_pipeline(spec: dict, seed: int, threads: int) -> Pipeline:
    """构建包含全部预处理的未拟合 Pipeline。"""
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if spec["family"] == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    transform = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL,
            ),
        ]
    )
    if spec["family"] == "dummy":
        model = DummyClassifier(strategy="most_frequent")
    elif spec["family"] == "logistic_regression":
        model = LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=seed, **spec["params"]
        )
    elif spec["family"] == "random_forest":
        model = RandomForestClassifier(
            random_state=seed,
            n_jobs=threads,
            max_features="sqrt",
            class_weight="balanced_subsample",
            **spec["params"],
        )
    else:
        raise ValueError(f"未知模型: {spec['family']}")
    return Pipeline([("preprocess", transform), ("classifier", model)])
