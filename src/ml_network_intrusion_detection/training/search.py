"""在测试评估前固定的有限搜索空间；不依赖历史测试指标生成候选。"""


def search_candidates(profile: str) -> list[dict]:
    """返回完整参数对照和 Transformer 候选；不将 smoke 轮数覆盖为正式轮数。"""

    def spec(name: str, family: str, **params) -> dict:
        return {"name": name, "family": family, "params": params}

    if profile == "transformer_smoke":
        return [
            spec("dummy", "dummy"),
            spec("linear-smoke", "linear"),
            spec("mlp-smoke", "mlp"),
            spec(
                "transformer-smoke", "transformer", d_model=16, n_heads=2, n_layers=1, dropout=0.1
            ),
        ]
    return [
        spec("dummy", "dummy"),
        spec("linear-original", "linear", epochs=20),
        spec("linear-lr003-e40", "linear", learning_rate=0.003, epochs=40),
        spec("linear-lr01-wd001-e40", "linear", learning_rate=0.01, weight_decay=0.001, epochs=40),
        spec("mlp-original", "mlp", hidden_dims=(128, 64), epochs=20),
        spec(
            "mlp128-lr0005-wd001-e40",
            "mlp",
            hidden_dims=(128, 64),
            learning_rate=0.0005,
            weight_decay=0.001,
            epochs=40,
        ),
        spec("mlp128-drop01-e40", "mlp", hidden_dims=(128, 64), dropout=0.1, epochs=40),
        spec("mlp256-drop01-e40", "mlp", hidden_dims=(256, 128), dropout=0.1, epochs=40),
        spec("mlp64-lr003-e40", "mlp", hidden_dims=(64, 32), learning_rate=0.003, epochs=40),
        spec(
            "transformer16-l1",
            "transformer",
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.1,
            epochs=15,
            batch_size=256,
        ),
        spec(
            "transformer32-l2",
            "transformer",
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            learning_rate=0.0003,
            epochs=15,
            batch_size=256,
        ),
    ]
