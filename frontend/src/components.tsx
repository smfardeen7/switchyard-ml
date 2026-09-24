import type { AuditEvent, Metrics, Model, Policy, Prediction, Sample } from './types';

export const percentage = (value: number) => `${(value * 100).toFixed(1)}%`;
export const milliseconds = (value: number) => `${value.toFixed(1)} ms`;
export const timestamp = (value: string) =>
  new Date(value).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
export const shortVersion = (value: string) => value.replace('digits-', '');
export function Icon({
  name,
}: {
  name: 'overview' | 'models' | 'traffic' | 'history' | 'arrow' | 'play' | 'shield';
}) {
  const paths = {
    overview: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
    models: 'M3 7l9-4 9 4-9 4-9-4z M3 12l9 4 9-4 M3 17l9 4 9-4',
    traffic: 'M4 6h8l4 4h4 M4 18h8l4-4h4 M17 7l3 3-3 3 M17 11l3 3-3 3',
    history: 'M3 12a9 9 0 1 0 3-7 M3 3v5h5 M12 7v5l3 2',
    arrow: 'M5 12h14 M13 6l6 6-6 6',
    play: 'M7 4l13 8-13 8z',
    shield: 'M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3z M8 12l3 3 5-6',
  };
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  );
}
export function MetricCard({
  label,
  value,
  detail,
  tone = '',
}: {
  label: string;
  value: string;
  detail: string;
  tone?: string;
}) {
  return (
    <div className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
export function TrafficChart({ metrics, policy }: { metrics: Metrics; policy: Policy }) {
  const points = metrics.timeseries;
  const width = 650,
    height = 134;
  const max =
    Math.max(policy.guardrails.max_p95_ms, ...points.map((point) => point.p95_ms), 1) * 1.15;
  const line = points
    .map(
      (point, index) =>
        `${points.length > 1 ? (index / (points.length - 1)) * width : width / 2},${height - (point.p95_ms / max) * height}`,
    )
    .join(' ');
  const area = points.length ? `0,${height} ${line} ${width},${height}` : '';
  const guardY = height - (policy.guardrails.max_p95_ms / max) * height;
  return (
    <section className="traffic-panel" id="traffic">
      <div className="panel-heading">
        <div>
          <span className="section-index">Request path</span>
          <h2>Traffic & response time</h2>
        </div>
        <span className="chart-period">{metrics.window_seconds}s window</span>
      </div>
      <div className="allocation-labels">
        <span>
          <i className="dot champion-dot" />
          Champion <strong>{percentage(1 - policy.weight)}</strong>
        </span>
        <span>
          <i className="dot canary-dot" />
          Canary <strong>{percentage(policy.weight)}</strong>
        </span>
      </div>
      <div
        className="allocation"
        role="img"
        aria-label={`Configured traffic: champion ${percentage(1 - policy.weight)}, canary ${percentage(policy.weight)}`}
      >
        <span style={{ width: `${(1 - policy.weight) * 100}%` }} />
        <span style={{ width: `${policy.weight * 100}%` }} />
      </div>
      <div className="allocation-versions">
        <code>{shortVersion(policy.champion)}</code>
        <code>{policy.canary ? shortVersion(policy.canary) : 'No active canary'}</code>
      </div>
      <div className="chart-header">
        <span>End-to-end p95</span>
        <strong>{milliseconds(metrics.aggregate.p95_ms)}</strong>
      </div>
      {points.length ? (
        <svg
          className="latency-chart"
          viewBox={`0 0 ${width} ${height + 22}`}
          role="img"
          aria-label={`Observed p95 latency across ${points.length} time buckets; latest aggregate ${milliseconds(metrics.aggregate.p95_ms)}`}
        >
          {[0, 0.5, 1].map((value) => (
            <line
              key={value}
              x1="0"
              x2={width}
              y1={value * height}
              y2={value * height}
              stroke="#ffffff12"
            />
          ))}
          <line x1="0" x2={width} y1={guardY} y2={guardY} stroke="#d6ad62" strokeDasharray="5 5" />
          <text
            x={width - 3}
            y={Math.max(10, guardY - 5)}
            textAnchor="end"
            fill="#d6ad62"
            fontSize="10"
          >
            Canary limit {policy.guardrails.max_p95_ms} ms
          </text>
          <polygon points={area} fill="#4bd5cf15" />
          <polyline points={line} fill="none" stroke="#4bd5cf" strokeWidth="2.5" />
          <text x="0" y={height + 19} fill="#97b0bc" fontSize="10">
            {new Date(points[0].at).toLocaleTimeString()}
          </text>
          <text x={width} y={height + 19} textAnchor="end" fill="#97b0bc" fontSize="10">
            {new Date(points[points.length - 1].at).toLocaleTimeString()}
          </text>
        </svg>
      ) : (
        <div className="chart-empty">
          No observations in this window. Send requests to populate the chart.
        </div>
      )}
      <p className="chart-note">
        {metrics.retention?.latency_samples_truncated &&
          `Latency percentiles use the latest ${metrics.retention.latency_samples.toLocaleString()} retained observations (capacity ${metrics.retention.latency_sample_capacity.toLocaleString()}); throughput and counts include all ${metrics.retention.window_requests.toLocaleString()} requests in the window. `}
        Configured allocation uses request routing keys. The chart shows all traffic; automatic
        rollback evaluates canary observations only.
      </p>
    </section>
  );
}
export function VersionTable({ metrics, policy }: { metrics: Metrics; policy: Policy }) {
  return (
    <div className="table-wrap">
      <table>
        <caption>Observed per-version performance</caption>
        <thead>
          <tr>
            <th>Version</th>
            <th>Requests</th>
            <th>Error rate</th>
            <th>p95</th>
            <th>Queue p50</th>
            <th>Inference p50</th>
          </tr>
        </thead>
        <tbody>
          {metrics.by_version.map((model) => (
            <tr key={model.version}>
              <th>
                <i
                  className={`dot ${model.version === policy.champion ? 'champion-dot' : 'canary-dot'}`}
                />
                {shortVersion(model.version)}
              </th>
              <td>{model.requests.toLocaleString()}</td>
              <td>{percentage(model.error_rate)}</td>
              <td>{milliseconds(model.p95_ms)}</td>
              <td>{milliseconds(model.queue_ms)}</td>
              <td>{milliseconds(model.inference_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!metrics.by_version.length && <p className="empty">No per-version observations yet.</p>}
    </div>
  );
}
export function Registry({ models, policy }: { models: Model[]; policy: Policy }) {
  return (
    <section className="panel" id="registry">
      <div className="panel-heading">
        <div>
          <span className="section-index">Artifact registry</span>
          <h2>Versions you can verify</h2>
        </div>
        <span className="subtle">{models.length} local models</span>
      </div>
      <div className="model-grid">
        {models.map((model) => (
          <article className="model-card" key={model.version}>
            <div className="model-top">
              <span
                className={`role-badge ${policy.champion === model.version ? 'champion' : policy.canary === model.version ? 'canary' : ''}`}
              >
                {policy.champion === model.version
                  ? 'Champion'
                  : policy.canary === model.version
                    ? 'Canary'
                    : 'Candidate'}
              </span>
              <span>ONNX · CPU</span>
            </div>
            <h3>{model.title}</h3>
            <code>{model.version}</code>
            <p>{model.description}</p>
            <dl className="model-stats">
              <div>
                <dt>Holdout accuracy</dt>
                <dd>{percentage(model.accuracy)}</dd>
              </div>
              <div>
                <dt>Test / train samples</dt>
                <dd>
                  {model.test_samples} / {model.train_samples}
                </dd>
              </div>
            </dl>
            <details>
              <summary>
                <span>
                  SHA-256 <code>{model.sha256.slice(0, 12)}…</code>
                </span>
                <span>Inspect</span>
              </summary>
              <code className="checksum">{model.sha256}</code>
              <p>
                {(model.artifact_bytes / 1024).toFixed(1)} KiB artifact · {model.framework}
                <br />
                Created {timestamp(model.created_at)}
              </p>
            </details>
          </article>
        ))}
      </div>
    </section>
  );
}
export function Digit({ sample }: { sample: Sample }) {
  return (
    <svg
      viewBox="0 0 8 8"
      aria-label={`Digit sample with reference label ${sample.label}`}
      role="img"
      className="digit-grid"
    >
      {sample.pixels.map((value, index) => (
        <rect
          key={index}
          x={index % 8}
          y={Math.floor(index / 8)}
          width="1"
          height="1"
          fill={`rgb(${Math.round((value / 16) * 255)} ${Math.round((value / 16) * 255)} ${Math.round((value / 16) * 255)})`}
        />
      ))}
    </svg>
  );
}
export function PredictionResult({ prediction }: { prediction: Prediction | null }) {
  if (!prediction)
    return (
      <div className="prediction-empty">
        Run a sample to inspect the prediction, serving version and timing breakdown.
      </div>
    );
  return (
    <div className="prediction-result" aria-live="polite">
      <div className="prediction-answer">
        <span>
          Predicted digit<strong>{prediction.prediction}</strong>
        </span>
        <div>
          <strong>{percentage(prediction.probabilities[prediction.prediction])}</strong>
          <small>model probability</small>
          <code>{shortVersion(prediction.version)}</code>
        </div>
      </div>
      <div className="prediction-timings">
        <span>
          Total <b>{milliseconds(prediction.latency_ms)}</b>
        </span>
        <span>
          Queue <b>{milliseconds(prediction.queue_ms)}</b>
        </span>
        <span>
          Inference <b>{milliseconds(prediction.inference_ms)}</b>
        </span>
        <span>
          Batch <b>{prediction.batch_size}</b>
        </span>
      </div>
      <span className="request-id">
        Request {prediction.request_id} · revision {prediction.policy_revision}
      </span>
    </div>
  );
}
export function EventTimeline({ events }: { events: AuditEvent[] }) {
  return (
    <section className="panel events-panel" id="events">
      <div className="panel-heading">
        <div>
          <span className="section-index">Audit trail</span>
          <h2>Every release leaves a record</h2>
        </div>
        <span className="subtle">Newest first</span>
      </div>
      {events.length ? (
        <ol className="timeline">
          {events.map((event) => (
            <li key={event.id} className={event.type.includes('rollback') ? 'rollback-event' : ''}>
              <span className="event-dot" />
              <div>
                <div className="event-top">
                  <strong>{event.type.replaceAll('_', ' ')}</strong>
                  <time dateTime={event.at}>{timestamp(event.at)}</time>
                </div>
                <p>{event.message}</p>
                <details>
                  <summary>Revision {event.revision} · event details</summary>
                  <pre>{JSON.stringify(event.details, null, 2)}</pre>
                </details>
              </div>
            </li>
          ))}
        </ol>
      ) : (
        <p className="empty">
          No release events yet. Start a canary to create the first audit record.
        </p>
      )}
    </section>
  );
}
