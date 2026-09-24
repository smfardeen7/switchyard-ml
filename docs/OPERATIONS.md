# Operating the reference deployment

Switchyard deliberately runs one API process and one controller. SQLite contains release policy and audit history; queue state, request windows and Prometheus counters are process-local. An active canary is rolled back on restart because its observation window cannot safely be resumed. Do not increase Uvicorn workers or Kubernetes replicas without first externalizing policy coordination, routing state and observations.

## Local containers

```sh
docker compose up --build --detach --wait
```

The console is at `http://127.0.0.1:8080`, the API at `http://127.0.0.1:8090`, and Prometheus at `http://127.0.0.1:9090`. Compose explicitly enables **demo mode** and the local-only key `local-development-only`. A named volume retains models and controller state. The API image runs as UID 10001; its lifespan trains/exports missing reference models, validates checksums, and warms ONNX sessions before `/ready` succeeds.

```sh
docker compose logs --follow api
docker compose down
```

`down` retains the named volume. Stop the API before copying its SQLite state for a consistent offline backup. Restore both model artifacts and release data from the same stopped instance. Delete the named volume only when deliberately resetting the demo.

For production mode, set `SWITCHYARD_ENV=production` and provide a randomly generated `SWITCHYARD_API_KEY` of at least 32 characters. Do not reuse the demo key. Production mode disables fault injection. Inference and mutation requests require the bearer key; read-only model metadata and metrics expose the public reference dataset only. TLS termination and external access control are deployment responsibilities, not implemented by this local Compose reference.

## Kubernetes example

`deploy/kubernetes.yaml` is a single-instance reference with a PVC, non-root UID, CPU/memory limits, startup/readiness/liveness probes and a `Recreate` strategy. It does not claim multi-replica availability. The example uses `switchyard:local`; build and load that image into a local cluster, or substitute an image pushed to your registry.

```sh
docker build -t switchyard:local .
# For a kind cluster:
kind load docker-image switchyard:local
kubectl create secret generic switchyard-api --from-literal=token="$(openssl rand -hex 32)"
kubectl apply -f deploy/kubernetes.yaml
kubectl rollout status deployment/switchyard
kubectl port-forward service/switchyard 8090:8090
```

A default storage class must satisfy the PVC. Prometheus scrape annotations are hints for an existing cluster monitoring setup, not an installed monitoring stack. The reference manifest starts in production mode and cannot run the injected-fault recording scenario.

## Evidence and health

- `/ready`: warmed model runtime. `/health`: API process health.
- `/metrics`: Prometheus exposition; `/v1/metrics`: rolling UI observations.
- `/v1/events`: release/rollback audit history including guardrail evidence.
- `scripts/record_demo.py`: actual local HTTP traffic, controlled canary faults, automatic rollback assertion and successful recovery check. Requires demo mode and changes local rollout state.

```sh
.venv/bin/python scripts/record_demo.py
```

The public Pages site builds with `VITE_DEMO_MODE=1` and displays the checked-in captured experiment. It has no live API and disables deployment controls. CI tests the Python code, builds TypeScript, runs a real engine benchmark smoke, then builds/starts Compose and records an HTTP rollback experiment. Pages deploys only after those jobs pass on `main`.

To reproduce the public recording without earlier traffic or events, use a dedicated API process with a fresh `SWITCHYARD_DATA_DIR`, then pass its URL via `scripts/record_demo.py --url http://127.0.0.1:8091`. Stop that process when finished. Reusing an existing process retains its rolling metrics and event history; the recorder reports what the API actually observed.

## Validation limits

Local Docker availability and Kubernetes execution must be stated separately from Python/HTTP verification. The manifests are examples until an actual deployment is exercised. Passing the CI Compose job establishes container build/start and local networking behavior; it does not establish cloud availability, autoscaling, GPU performance or a production SLO.
