import json

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from switchyard.models import ModelRegistry, bootstrap


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    bootstrap(path)
    return path


@pytest.fixture(scope="module")
def registry(artifacts):
    return ModelRegistry(artifacts)


def test_real_onnx_predictions_match_independently_fitted_sklearn(registry):
    digits = load_digits()
    train, holdout = train_test_split(
        np.arange(len(digits.target)),
        test_size=0.25,
        random_state=42,
        stratify=digits.target,
    )
    estimators = {
        "digits-logreg-v1": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        ),
        "digits-rf-v2": RandomForestClassifier(
            n_estimators=96, random_state=42, n_jobs=1
        ),
    }
    for version, estimator in estimators.items():
        estimator.fit(digits.data[train], digits.target[train])
        probabilities = registry.predict(
            version, digits.data[holdout].astype(np.float32)
        )
        np.testing.assert_allclose(
            probabilities,
            estimator.predict_proba(digits.data[holdout]),
            rtol=1e-4,
            atol=1e-5,
        )
        assert probabilities.shape == (450, 10)
        accuracy = (probabilities.argmax(axis=1) == digits.target[holdout]).mean()
        assert accuracy >= 0.9
        assert next(
            model for model in registry.list_models() if model["version"] == version
        )["accuracy"] == pytest.approx(accuracy)


def test_registry_metadata_and_samples_are_valid_and_not_mutable(registry):
    models = registry.list_models()
    assert {m["version"] for m in models} == {"digits-logreg-v1", "digits-rf-v2"}
    for model in models:
        assert model["framework"] == "onnxruntime"
        assert model["train_samples"] == 1347
        assert model["test_samples"] == 450
        assert len(model["sha256"]) == 64
        assert model["artifact_bytes"] > 1000
    models[0]["accuracy"] = -1
    assert registry.list_models()[0]["accuracy"] >= 0
    samples = registry.sample_data()
    assert len(samples) >= 10
    assert {sample["label"] for sample in samples} == set(range(10))
    assert all(len(sample["pixels"]) == 64 for sample in samples)
    samples[0]["pixels"][0] = 999
    assert registry.sample_data()[0]["pixels"][0] <= 16


def test_bootstrap_preserves_existing_artifacts(artifacts):
    before = {
        path.name: path.read_bytes() for path in artifacts.iterdir() if path.is_file()
    }
    result = bootstrap(artifacts)
    assert result["models"]
    assert before == {
        path.name: path.read_bytes() for path in artifacts.iterdir() if path.is_file()
    }


def test_corrupt_artifact_rejected_before_any_session_is_created(
    artifacts, tmp_path, monkeypatch
):
    import shutil
    import switchyard.models as models

    for path in artifacts.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    model = next(tmp_path.glob("*.onnx"))
    model.write_bytes(model.read_bytes()[:-1] + b"!")
    sessions = []
    original = models.ort.InferenceSession

    def observe(*args, **kwargs):
        sessions.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(models.ort, "InferenceSession", observe)
    with pytest.raises(ValueError, match="checksum"):
        ModelRegistry(tmp_path)
    assert sessions == []


@pytest.mark.parametrize(
    "field,value",
    [("accuracy", float("nan")), ("artifact", "../escape.onnx"), ("test_samples", 0)],
)
def test_invalid_metadata_rejected(artifacts, tmp_path, field, value):
    import shutil

    for path in artifacts.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    manifest = json.loads((tmp_path / "registry.json").read_text())
    manifest["models"][0][field] = value
    (tmp_path / "registry.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ModelRegistry(tmp_path)


@pytest.mark.parametrize(
    "rows",
    [
        np.zeros((1, 63)),
        np.zeros(64),
        np.empty((0, 64)),
        np.full((1, 64), np.nan),
        np.full((1, 64), 17),
        np.full((1, 64), -1),
    ],
)
def test_registry_rejects_invalid_input(registry, rows):
    with pytest.raises(ValueError):
        registry.predict("digits-logreg-v1", rows)


def test_unknown_version_raises_key_error(registry):
    with pytest.raises(KeyError):
        registry.predict("unknown", np.zeros((1, 64)))


def test_same_seed_reproduces_model_checksums_and_holdout(artifacts, tmp_path):
    previous = json.loads((artifacts / "registry.json").read_text())
    generated = bootstrap(tmp_path, seed=42)
    assert generated["split"] == previous["split"]
    assert [model["sha256"] for model in generated["models"]] == [
        model["sha256"] for model in previous["models"]
    ]
    assert (tmp_path / "samples.json").read_bytes() == (
        artifacts / "samples.json"
    ).read_bytes()


@pytest.mark.asyncio
async def test_batcher_runs_real_onnx_sessions_without_changing_predictions(registry):
    import asyncio
    from switchyard.batching import DynamicBatcher

    samples = registry.sample_data()[:12]
    rows = np.asarray([sample["pixels"] for sample in samples], dtype=np.float32)
    batcher = DynamicBatcher(registry, max_batch_size=4, max_wait_ms=10)
    await batcher.start()
    try:
        for version in registry.versions:
            expected = registry.predict(version, rows)
            results = await asyncio.gather(
                *(batcher.predict(version, row) for row in rows)
            )
            np.testing.assert_allclose(
                [result.probabilities for result in results],
                expected,
                rtol=1e-4,
                atol=1e-5,
            )
            assert all(result.batch_size == 4 for result in results)
    finally:
        await batcher.close()
