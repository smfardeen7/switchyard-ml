import { useEffect, useRef, useState } from 'react';
import { recordedMode, write } from './api';
import {
  Digit,
  EventTimeline,
  Icon,
  MetricCard,
  PredictionResult,
  Registry,
  TrafficChart,
  VersionTable,
  milliseconds,
  percentage,
  shortVersion,
  timestamp,
} from './components';
import { useConsoleData } from './useConsoleData';
import type { Prediction } from './types';

export default function App() {
  const { data, recording, frameIndex, setFrameIndex, error, updatedAt, refresh } =
    useConsoleData();
  const [token, setToken] = useState('local-development-only');
  const [candidate, setCandidate] = useState('');
  const [weight, setWeight] = useState(0.25);
  const [faultRate, setFaultRate] = useState(50);
  const [faultLatency, setFaultLatency] = useState(0);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState<{ message: string; error: boolean } | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [progress, setProgress] = useState<{
    completed: number;
    success: number;
    failed: number;
  } | null>(null);
  const operations = useRef<AbortController | null>(null);
  useEffect(() => () => operations.current?.abort(), []);
  const policy = data?.policy;
  const currentCandidate =
    data?.models.find((model) => model.version === candidate && model.version !== policy?.champion)
      ?.version ||
    data?.models.find((model) => model.version !== policy?.champion)?.version ||
    '';
  const sampleOptions =
    data?.samples
      .filter(
        (sample, index, samples) =>
          samples.findIndex((item) => item.label === sample.label) === index,
      )
      .slice(0, 10) || [];
  const sample = sampleOptions[sampleIndex] || sampleOptions[0];
  const canaryMetrics = data?.metrics.by_version.find((model) => model.version === policy?.canary);
  const displayedWeight = recordedMode || policy?.canary ? (policy?.weight ?? 0) : weight;
  const canWrite = !recordedMode && !busy && !error;
  const championModel = data?.models.find((model) => model.version === policy?.champion);
  const candidateModel = data?.models.find((model) => model.version === currentCandidate);
  const qualityPass =
    championModel && candidateModel
      ? candidateModel.accuracy >= championModel.accuracy - 0.03
      : false;

  async function action(label: string, path: string, body: unknown) {
    if (recordedMode || busy) return;
    setBusy(label);
    setNotice(null);
    try {
      await write(path, body, token);
      setNotice({ message: `${label} accepted by the API.`, error: false });
      refresh();
    } catch (error) {
      setNotice({
        message: error instanceof Error ? error.message : 'Request failed.',
        error: true,
      });
      refresh();
    } finally {
      setBusy('');
    }
  }
  async function infer() {
    if (!sample || recordedMode || busy) return;
    setBusy('Running inference');
    setNotice(null);
    setPrediction(null);
    const controller = new AbortController();
    operations.current = controller;
    try {
      setPrediction(
        await write<Prediction>(
          '/v1/predict',
          {
            pixels: sample.pixels,
            routing_key: `sample-${sample.id}-${Date.now()}`,
            timeout_ms: 2000,
          },
          token,
          controller.signal,
        ),
      );
      refresh();
    } catch (error) {
      setNotice({
        message: error instanceof Error ? error.message : 'Inference failed.',
        error: true,
      });
    } finally {
      setBusy('');
      operations.current = null;
    }
  }
  async function sendBurst() {
    if (!data?.samples.length || recordedMode || busy) return;
    setBusy('Sending requests');
    setNotice(null);
    setProgress({ completed: 0, success: 0, failed: 0 });
    const controller = new AbortController();
    operations.current = controller;
    let next = 0,
      success = 0,
      failed = 0;
    const burstId = Date.now();
    await Promise.all(
      Array.from({ length: 12 }, async () => {
        while (next < 120 && !controller.signal.aborted) {
          const index = next++;
          try {
            await write(
              '/v1/predict',
              {
                pixels: data.samples[index % data.samples.length].pixels,
                routing_key: `console-${burstId}-${index}`,
                timeout_ms: 2000,
              },
              token,
              controller.signal,
            );
            success++;
          } catch {
            failed++;
          }
          if (!controller.signal.aborted)
            setProgress({ completed: success + failed, success, failed });
        }
      }),
    );
    if (!controller.signal.aborted) {
      setNotice({
        message: `Completed 120 HTTP requests with 12 concurrent workers: ${success} succeeded, ${failed} failed. Inspect metrics and events for the observed outcome.`,
        error: failed === 120,
      });
      refresh();
    }
    setBusy('');
    operations.current = null;
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to console
      </a>
      <aside className="sidebar">
        <a href="#workspace" className="brand" aria-label="Switchyard console">
          <span className="switch-mark">
            <i />
            <i />
            <i />
          </span>
          <span>
            switchyard<small>Model serving console</small>
          </span>
        </a>
        <div className="sidebar-environment">
          <i className="dot" />
          {recordedMode ? 'Recorded experiment' : 'Local control plane'}
        </div>
        <nav aria-label="Console sections">
          <a href="#workspace">
            <Icon name="overview" />
            Overview
          </a>
          <a href="#traffic">
            <Icon name="traffic" />
            Traffic & metrics
          </a>
          <a href="#registry">
            <Icon name="models" />
            Model registry
          </a>
          <a href="#events">
            <Icon name="history" />
            Release history
          </a>
        </nav>
        <div className="sidebar-bottom">
          <span className="cpu-mark">CPU</span>
          <p>
            Small workload.
            <br />
            Real release mechanics.
          </p>
          <span>ONNX Runtime / digits</span>
          <a href="https://github.com/smfardeen7/switchyard-ml" target="_blank" rel="noreferrer">
            Source & runbook ↗
          </a>
        </div>
      </aside>
      <main id="workspace">
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Workspace <span>/</span> digits-classifier
            </div>
            <h1>Inference control plane</h1>
            <p>Route traffic. Observe the canary. Keep a way back.</p>
          </div>
          <div className="connection-state">
            <span
              className={`connection-badge ${recordedMode ? 'recorded' : error ? 'disconnected' : ''}`}
            >
              <i className="dot" />
              {recordedMode
                ? 'Recorded · read only'
                : error
                  ? 'Connection interrupted'
                  : data
                    ? 'API connected'
                    : 'Connecting'}
            </span>
            <small>
              {updatedAt
                ? `${recordedMode ? 'Captured' : 'Updated'} ${timestamp(updatedAt)}`
                : 'Waiting for observations'}
            </small>
          </div>
        </header>
        {recordedMode && (
          <section className="recording-banner" aria-label="Recorded experiment">
            <div>
              <strong>
                <Icon name="play" />
                An actual experiment, captured for inspection
              </strong>
              <p>
                This page replays recorded API state. It is not a running inference service. Start
                the repository locally to operate the controls.
              </p>
              {recording && (
                <span className="capture-source">
                  Source: {recording.source} · captured {timestamp(recording.captured_at)}
                </span>
              )}
            </div>
            {recording && (
              <label>
                Inspect a captured frame
                <select
                  value={frameIndex}
                  onChange={(event) => {
                    setFrameIndex(Number(event.target.value));
                    setPrediction(null);
                  }}
                >
                  {recording.frames.map((frame, index) => (
                    <option key={index} value={index}>
                      {index + 1}. {frame.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </section>
        )}
        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>
                {data ? 'Displayed observations may be stale.' : 'Console data is unavailable.'}
              </strong>
              <p>
                {error}{' '}
                {!recordedMode && 'Start the API on port 8090; live reads retry every two seconds.'}
              </p>
            </div>
            <button onClick={refresh}>Retry now</button>
          </div>
        )}
        {!data ? (
          <div className="loading-panel" role="status">
            <span className="loading-symbol">╱╲</span>
            <h2>
              {error
                ? 'Waiting for a valid data source'
                : recordedMode
                  ? 'Opening the experiment recording'
                  : 'Connecting to the serving runtime'}
            </h2>
            <p>
              {recordedMode
                ? 'The recording is generated from a real integration run. No placeholder measurements are shown.'
                : 'Models must be verified and warmed before metrics and controls appear.'}
            </p>
          </div>
        ) : (
          <>
            <div className="workspace-meta">
              <span>
                <i className="dot champion-dot" />
                Champion <code>{shortVersion(data.policy.champion)}</code>
              </span>
              <span>
                Revision <b>{data.policy.revision}</b>
              </span>
              <span className={`policy-status status-${data.policy.status}`}>
                {data.policy.status.replaceAll('_', ' ')}
              </span>
              <span className="frame-label">
                {recordedMode ? data.label : 'One process · CPU reference workload'}
              </span>
            </div>
            <div className="metric-grid">
              <MetricCard
                label="Total requests"
                value={data.metrics.totals.requests.toLocaleString()}
                detail={`${data.metrics.totals.rejections} rejected at admission`}
              />
              <MetricCard
                label="Throughput"
                value={`${data.metrics.aggregate.throughput_rps.toFixed(1)}`}
                detail={`requests / sec · ${data.metrics.window_seconds}s window`}
              />
              <MetricCard
                label="End-to-end p95"
                value={milliseconds(data.metrics.aggregate.p95_ms)}
                detail={`p99 ${milliseconds(data.metrics.aggregate.p99_ms)}`}
              />
              <MetricCard
                label="Observed error rate"
                value={percentage(data.metrics.aggregate.error_rate)}
                detail={`${data.metrics.totals.errors} total request errors`}
                tone={data.metrics.aggregate.error_rate > 0 ? 'has-errors' : ''}
              />
            </div>
            <div className="operations-grid">
              <div className="observability">
                <TrafficChart metrics={data.metrics} policy={data.policy} />
                <VersionTable metrics={data.metrics} policy={data.policy} />
                <div className="queue-strip">
                  <Icon name="models" />
                  <strong>Bounded batching</strong>
                  <span>
                    Queued <b>{data.metrics.queue.queue_depth}</b>
                  </span>
                  <span>
                    In flight <b>{data.metrics.queue.inflight}</b>
                  </span>
                  <span>
                    Batch cap <b>{data.metrics.queue.max_batch_size}</b>
                  </span>
                  <span>
                    Queue cap <b>{data.metrics.queue.queue_capacity_per_version}</b> / version
                  </span>
                </div>
              </div>
              <section className="release-panel" id="releases">
                <div className="panel-heading">
                  <div>
                    <span className="section-index">Release controls</span>
                    <h2>A guarded route forward</h2>
                  </div>
                  <Icon name="shield" />
                </div>
                {recordedMode && (
                  <div className="readonly-note">
                    Read-only recording. Live actions are disabled.
                  </div>
                )}
                <label>
                  Candidate version
                  <select
                    value={currentCandidate}
                    onChange={(event) => setCandidate(event.target.value)}
                    disabled={!canWrite || !!data.policy.canary}
                  >
                    {data.models
                      .filter((model) => model.version !== data.policy.champion)
                      .map((model) => (
                        <option key={model.version} value={model.version}>
                          {shortVersion(model.version)}
                        </option>
                      ))}
                  </select>
                </label>
                <div className="range-heading">
                  <label htmlFor="weight">
                    {recordedMode
                      ? 'Captured canary allocation'
                      : data.policy.canary
                        ? 'Active canary allocation'
                        : 'New canary allocation'}
                  </label>
                  <strong>{Math.round(displayedWeight * 100)}%</strong>
                </div>
                <input
                  id="weight"
                  className="weight-range"
                  type="range"
                  min={recordedMode ? 0 : 0.05}
                  max=".5"
                  step=".05"
                  value={displayedWeight}
                  disabled={!canWrite || !!data.policy.canary}
                  onChange={(event) => setWeight(Number(event.target.value))}
                />
                <div className="range-ends">
                  <span>{recordedMode ? '0%' : '5%'}</span>
                  <span>50%</span>
                </div>
                <button
                  className="primary-button"
                  disabled={!canWrite || !!data.policy.canary || !qualityPass}
                  onClick={() =>
                    void action('Start canary', '/v1/rollouts', {
                      candidate: currentCandidate,
                      weight,
                      expected_revision: data.policy.revision,
                    })
                  }
                >
                  <Icon name="traffic" />
                  {busy === 'Start canary' ? 'Starting…' : 'Start canary'}
                </button>
                <p className={`control-help ${qualityPass ? '' : 'quality-fail'}`}>
                  {qualityPass
                    ? 'Quality gate: candidate is within 3 percentage points of champion holdout accuracy.'
                    : 'Quality gate blocks this candidate: accuracy is more than 3 percentage points below champion.'}
                </p>
                <dl className="guardrail-list">
                  <div>
                    <dt>Error-rate ceiling</dt>
                    <dd>{percentage(data.policy.guardrails.max_error_rate)}</dd>
                  </div>
                  <div>
                    <dt>Canary p95 ceiling</dt>
                    <dd>{data.policy.guardrails.max_p95_ms} ms</dd>
                  </div>
                  <div>
                    <dt>Minimum observations</dt>
                    <dd>{data.policy.guardrails.min_samples}</dd>
                  </div>
                  <div>
                    <dt>Observation window</dt>
                    <dd>{data.policy.guardrails.window_size} requests</dd>
                  </div>
                </dl>
                <div className="release-actions">
                  <button
                    disabled={!canWrite || !data.policy.canary}
                    onClick={() =>
                      void action('Promote candidate', '/v1/rollouts/promote', {
                        expected_revision: data.policy.revision,
                      })
                    }
                  >
                    Promote
                  </button>
                  <button
                    className="danger-outline"
                    disabled={!canWrite || !data.policy.canary}
                    onClick={() =>
                      void action('Rollback canary', '/v1/rollouts/rollback', {
                        expected_revision: data.policy.revision,
                        reason: 'Manual rollback from console',
                      })
                    }
                  >
                    Rollback
                  </button>
                </div>
                <p className="control-help">
                  Promotion is validated by the API against the current rollout’s observations.
                  {canaryMetrics
                    ? ` ${canaryMetrics.requests} canary requests appear in the displayed metrics window.`
                    : ''}
                </p>
                {data.policy.reason && <p className="policy-reason">{data.policy.reason}</p>}
              </section>
            </div>
            {notice && (
              <div
                className={`action-notice ${notice.error ? 'action-error' : ''}`}
                role={notice.error ? 'alert' : 'status'}
              >
                <span>{notice.message}</span>
                <button aria-label="Dismiss message" onClick={() => setNotice(null)}>
                  ×
                </button>
              </div>
            )}
            <section className="exercise-panel">
              <div>
                <span className="section-index">Exercise the release</span>
                <h2>Observe failure. Verify recovery.</h2>
                <p>
                  {recordedMode
                    ? 'These controls require the local live API. Inspect the recorded frames above to see the captured outcome.'
                    : 'Send actual digit requests, inject a canary-only fault, and inspect the persisted rollback event.'}
                </p>
              </div>
              <div className="exercise-actions">
                <button
                  className="request-button"
                  disabled={!canWrite || !data.samples.length}
                  onClick={() => void sendBurst()}
                >
                  <Icon name="play" />
                  {busy === 'Sending requests'
                    ? `Sending ${progress?.completed || 0} / 120`
                    : 'Send 120 requests'}
                  <small>12 concurrent workers</small>
                </button>
                <div className="chaos-controls">
                  <div className="chaos-fields">
                    <label>
                      Error rate %
                      <input
                        type="number"
                        min="0"
                        max="100"
                        step="5"
                        value={faultRate}
                        disabled={!canWrite || !data.demo_mode}
                        onChange={(event) =>
                          setFaultRate(Math.max(0, Math.min(100, Number(event.target.value))))
                        }
                      />
                    </label>
                    <label>
                      Added latency ms
                      <input
                        type="number"
                        min="0"
                        max="2000"
                        step="10"
                        value={faultLatency}
                        disabled={!canWrite || !data.demo_mode}
                        onChange={(event) =>
                          setFaultLatency(Math.max(0, Math.min(2000, Number(event.target.value))))
                        }
                      />
                    </label>
                  </div>
                  <button
                    className="chaos-button"
                    disabled={
                      !canWrite || !data.demo_mode || (!data.policy.canary && !data.chaos.enabled)
                    }
                    onClick={() =>
                      void action(
                        data.chaos.enabled ? 'Disable faults' : 'Enable canary faults',
                        '/v1/chaos',
                        {
                          enabled: !data.chaos.enabled,
                          error_rate: faultRate / 100,
                          latency_ms: faultLatency,
                        },
                      )
                    }
                  >
                    {data.chaos.enabled ? 'Disable active faults' : 'Inject canary faults'}
                  </button>
                  <small>
                    {recordedMode
                      ? 'Disabled in recorded mode'
                      : !data.demo_mode
                        ? 'Fault injection disabled in production'
                        : data.chaos.enabled
                          ? `Active: ${percentage(data.chaos.error_rate)} errors, ${data.chaos.latency_ms} ms latency`
                          : 'Demo mode only · active canary required'}
                  </small>
                </div>
              </div>
              {progress && (
                <div className="burst-progress" role="status">
                  <progress value={progress.completed} max="120" />
                  <span>
                    {progress.completed}/120 complete · {progress.success} successful ·{' '}
                    {progress.failed} failed
                  </span>
                </div>
              )}
            </section>
            <Registry models={data.models} policy={data.policy} />
            <div className="detail-grid">
              <section className="panel inference-panel" id="inference">
                <div className="panel-heading">
                  <div>
                    <span className="section-index">Reference workload</span>
                    <h2>Put a digit through the path</h2>
                  </div>
                  <span className="subtle">8 × 8 pixels</span>
                </div>
                <div className="sample-options" aria-label="Select a digit sample">
                  {sampleOptions.map((item, index) => (
                    <button
                      key={item.id}
                      className={sampleIndex === index ? 'selected' : ''}
                      aria-pressed={sampleIndex === index}
                      disabled={busy === 'Running inference'}
                      aria-label={`Sample ${item.id}, reference digit ${item.label}`}
                      onClick={() => {
                        setSampleIndex(index);
                        setPrediction(null);
                      }}
                    >
                      <Digit sample={item} />
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>
                {sample && (
                  <div className="sample-run">
                    <span>
                      Reference label <strong>{sample.label}</strong>
                      <small>64 finite values, range 0–16</small>
                    </span>
                    <button disabled={!canWrite} onClick={() => void infer()}>
                      {busy === 'Running inference' ? 'Running…' : 'Run inference'}
                      <Icon name="arrow" />
                    </button>
                  </div>
                )}
                {recordedMode ? (
                  <div className="prediction-empty">
                    Sample inputs are included in this recording. Inference requires the live local
                    API; this page does not generate predictions.
                  </div>
                ) : (
                  <PredictionResult prediction={prediction} />
                )}
              </section>
              <EventTimeline events={data.events} />
            </div>
            {!recordedMode && (
              <section className="auth-panel">
                <div>
                  <strong>Session authorization</strong>
                  <p>
                    Required for inference and release changes. Kept in memory only; never stored in
                    the browser.
                  </p>
                </div>
                <label htmlFor="token">
                  API bearer token
                  <input
                    id="token"
                    type="password"
                    autoComplete="off"
                    spellCheck="false"
                    value={token}
                    onChange={(event) => setToken(event.target.value)}
                  />
                </label>
              </section>
            )}
          </>
        )}
        <footer>
          <span>Switchyard · single-instance model serving reference</span>
          <span>
            {recordedMode
              ? 'Captured measurements. No live traffic.'
              : 'Real ONNX inference. No external model APIs.'}
          </span>
        </footer>
      </main>
    </div>
  );
}
