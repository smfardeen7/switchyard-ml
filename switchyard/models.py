"""Reproducible local training and a checksum-verified, warmed ONNX registry.

Training imports are deliberately lazy: serving never deserializes sklearn or
pickle objects. A registry is published last, after both model files and sample
data have been written. Files described by an existing registry are immutable.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import onnxruntime as ort

# Keep this self-contained reference service from emitting ORT usage events.
# ORT 1.30's native import may still create its timestamp/UUID .ses sidecar
# before this supported API can run; that runtime-owned file is not an artifact.
ort.disable_telemetry_events()

PUBLIC_FIELDS = (
    "version",
    "title",
    "framework",
    "accuracy",
    "test_samples",
    "train_samples",
    "sha256",
    "artifact_bytes",
    "created_at",
    "description",
)


def _write_atomic(path: Path, payload: bytes):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read valid JSON from {Path(path).name}") from exc


def _session(payload):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        payload, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _probability_output(session):
    inputs = session.get_inputs()
    if (
        len(inputs) != 1
        or inputs[0].type != "tensor(float)"
        or len(inputs[0].shape) != 2
        or inputs[0].shape[1] != 64
    ):
        raise ValueError("ONNX model must accept one float32 [batch,64] tensor")
    outputs = [
        output.name
        for output in session.get_outputs()
        if output.type == "tensor(float)"
        and len(output.shape) == 2
        and output.shape[1] == 10
    ]
    if len(outputs) != 1:
        raise ValueError(
            "ONNX model must expose one float32 [batch,10] probability tensor (ZipMap disabled)"
        )
    return inputs[0].name, outputs[0]


def _validate_rows(rows):
    try:
        rows = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Digit inputs must be numeric arrays of shape [batch,64]"
        ) from exc
    if rows.ndim != 2 or rows.shape[1] != 64 or rows.shape[0] == 0:
        raise ValueError(
            "Digit inputs must have shape [batch,64] with a non-empty batch"
        )
    if not np.isfinite(rows).all() or np.any(rows < 0) or np.any(rows > 16):
        raise ValueError("Digit pixels must be finite and between 0 and 16")
    return np.ascontiguousarray(rows)


def _validate_probabilities(probabilities, count):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (count, 10) or not np.isfinite(probabilities).all():
        raise ValueError("Model produced malformed probability output")
    if (
        np.any(probabilities < -1e-6)
        or np.any(probabilities > 1 + 1e-6)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Model output is not a normalized probability distribution")
    # ONNX tree sums can differ from one by a few float32 ulps.
    probabilities = np.clip(probabilities, 0, 1)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def bootstrap(root: Path, seed=42) -> dict:
    """Train/export the bundled digits reference workload into an empty registry.

    Calling this again returns the existing manifest without modifying artifacts.
    ModelRegistry performs integrity checks before those artifacts can be served.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "registry.json").exists():
        return _read_json(root / "registry.json")

    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import sklearn
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    digits = load_digits()
    train, test = train_test_split(
        np.arange(len(digits.target)),
        test_size=0.25,
        random_state=seed,
        stratify=digits.target,
    )
    classifiers = [
        (
            "digits-logreg-v1",
            "Scaled logistic regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            ),
            "StandardScaler and multinomial logistic regression; 64 grayscale pixel features.",
        ),
        (
            "digits-rf-v2",
            "Random forest",
            RandomForestClassifier(n_estimators=96, random_state=seed, n_jobs=1),
            "96 seeded decision trees on the same stratified training partition.",
        ),
    ]
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "dataset": "sklearn.datasets.load_digits",
        "training_library_version": sklearn.__version__,
        "split": {
            "method": "stratified_holdout",
            "test_fraction": 0.25,
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
        },
        "models": [],
    }
    created_at = datetime.now(timezone.utc).isoformat()
    for version, title, estimator, description in classifiers:
        estimator.fit(digits.data[train], digits.target[train])
        classifier = (
            estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        )
        graph = convert_sklearn(
            estimator,
            initial_types=[("pixels", FloatTensorType([None, 64]))],
            options={id(classifier): {"zipmap": False}},
            target_opset=17,
        )
        # The converter supplies a random graph identifier; normalize it so the
        # same seed and dependency versions produce the same artifact checksum.
        graph.graph.name = version
        payload = graph.SerializeToString()
        session = _session(payload)
        input_name, output_name = _probability_output(session)
        probabilities = _validate_probabilities(
            session.run(
                [output_name], {input_name: digits.data[test].astype(np.float32)}
            )[0],
            len(test),
        )
        reference = estimator.predict_proba(digits.data[test])
        if not np.allclose(probabilities, reference, rtol=1e-4, atol=1e-5):
            raise ValueError(f"ONNX parity check failed for {version}")
        _write_atomic(root / f"{version}.onnx", payload)
        manifest["models"].append(
            {
                "version": version,
                "title": title,
                "framework": "onnxruntime",
                "accuracy": float(
                    np.mean(probabilities.argmax(axis=1) == digits.target[test])
                ),
                "test_samples": len(test),
                "train_samples": len(train),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_bytes": len(payload),
                "created_at": created_at,
                "description": description,
                "artifact": f"{version}.onnx",
            }
        )
    # Three deterministic held-out examples of each class, never training rows.
    selected = [
        int(index)
        for label in range(10)
        for index in test[digits.target[test] == label][:3]
    ]
    samples = [
        {
            "id": f"digits-{index}",
            "label": int(digits.target[index]),
            "pixels": digits.data[index].astype(int).tolist(),
        }
        for index in selected
    ]
    _write_atomic(
        root / "samples.json", (json.dumps(samples, indent=2) + "\n").encode()
    )
    _write_atomic(
        root / "registry.json", (json.dumps(manifest, indent=2) + "\n").encode()
    )
    return manifest


class ModelRegistry:
    """Read-only ONNX sessions, exposed only after all checks and warmups pass."""

    def __init__(self, root: Path):
        root = Path(root)
        manifest = _read_json(root / "registry.json")
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported registry schema")
        entries = manifest.get("models")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
            raise ValueError("Registry must contain between 1 and 32 models")
        verified = []
        versions = set()
        # Verify every artifact first, so a broken registry starts no sessions.
        for entry in entries:
            if not isinstance(entry, dict) or not all(
                field in entry for field in (*PUBLIC_FIELDS, "artifact")
            ):
                raise ValueError("Model metadata is incomplete")
            version = entry["version"]
            if (
                not isinstance(version, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", version)
                or version in versions
            ):
                raise ValueError("Model versions must be unique lowercase slugs")
            versions.add(version)
            if entry["artifact"] != f"{version}.onnx":
                raise ValueError(
                    "Model artifact must be the version's local ONNX basename"
                )
            for field in ("test_samples", "train_samples", "artifact_bytes"):
                if (
                    not isinstance(entry[field], int)
                    or isinstance(entry[field], bool)
                    or entry[field] <= 0
                ):
                    raise ValueError(f"Model {field} must be a positive integer")
            accuracy = entry["accuracy"]
            if (
                not isinstance(accuracy, (float, int))
                or isinstance(accuracy, bool)
                or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1
            ):
                raise ValueError("Model accuracy must be a finite fraction")
            if entry["framework"] != "onnxruntime" or not all(
                isinstance(entry[field], str) and entry[field]
                for field in ("title", "description", "created_at", "sha256")
            ):
                raise ValueError("Model metadata has invalid descriptive fields")
            try:
                if datetime.fromisoformat(entry["created_at"]).tzinfo is None:
                    raise ValueError("timestamp needs a timezone")
            except ValueError as exc:
                raise ValueError(
                    "Model created_at must be an ISO timestamp with timezone"
                ) from exc
            if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ValueError("Model checksum must be a SHA-256 hex digest")
            path = root / entry["artifact"]
            try:
                if (
                    path.is_symlink()
                    or path.stat().st_size != entry["artifact_bytes"]
                    or entry["artifact_bytes"] > 128 * 1024 * 1024
                ):
                    raise ValueError(
                        f"Model artifact size or local-file integrity mismatch: {version}"
                    )
                payload = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"Cannot read model artifact: {version}") from exc
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Model checksum mismatch: {version}")
            verified.append((deepcopy(entry), payload))

        self._samples = _read_json(root / "samples.json")
        if not isinstance(self._samples, list) or not self._samples:
            raise ValueError("Registry sample data must be a non-empty list")
        ids = set()
        for sample in self._samples:
            if (
                not isinstance(sample, dict)
                or not isinstance(sample.get("id"), str)
                or not sample["id"]
                or sample["id"] in ids
            ):
                raise ValueError("Registry samples need unique identifiers")
            ids.add(sample["id"])
            if (
                not isinstance(sample.get("label"), int)
                or isinstance(sample["label"], bool)
                or not 0 <= sample["label"] <= 9
            ):
                raise ValueError("Registry sample labels must be digits")
            _validate_rows([sample.get("pixels", [])])
        self._metadata = {
            entry["version"]: {field: entry[field] for field in PUBLIC_FIELDS}
            for entry, _ in verified
        }
        self._sessions = {}
        for entry, payload in verified:
            try:
                session = _session(payload)
                input_name, output_name = _probability_output(session)
                self._sessions[entry["version"]] = (session, input_name, output_name)
                self.predict(
                    entry["version"],
                    np.array([self._samples[0]["pixels"]], dtype=np.float32),
                )
            except Exception as exc:
                raise ValueError(
                    f"Model session validation/warmup failed: {entry['version']}"
                ) from exc

    @property
    def versions(self) -> list[str]:
        return list(self._metadata)

    def list_models(self) -> list[dict]:
        return deepcopy(list(self._metadata.values()))

    def sample_data(self) -> list[dict]:
        return deepcopy(self._samples)

    def predict(self, version: str, rows: np.ndarray) -> np.ndarray:
        session, input_name, output_name = self._sessions[version]
        rows = _validate_rows(rows)
        return _validate_probabilities(
            session.run([output_name], {input_name: rows})[0], len(rows)
        )
