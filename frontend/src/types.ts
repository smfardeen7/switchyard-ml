export type Model = {
  version: string;
  title: string;
  framework: string;
  accuracy: number;
  test_samples: number;
  train_samples: number;
  sha256: string;
  artifact_bytes: number;
  created_at: string;
  description: string;
};
export type Sample = { id: string; label: number; pixels: number[] };
export type Guardrails = {
  max_error_rate: number;
  max_p95_ms: number;
  min_samples: number;
  window_size: number;
};
export type Policy = {
  revision: number;
  champion: string;
  canary: string | null;
  weight: number;
  status: 'inactive' | 'running' | 'rolled_back' | 'promoted';
  rollout_id: string | null;
  guardrails: Guardrails;
  updated_at: string;
  reason: string | null;
};
export type AuditEvent = {
  id: number;
  at: string;
  type: string;
  message: string;
  revision: number;
  details: Record<string, unknown>;
};
export type VersionMetrics = {
  version: string;
  requests: number;
  errors: number;
  error_rate: number;
  p50_ms: number;
  p95_ms: number;
  queue_ms: number;
  inference_ms: number;
};
export type Metrics = {
  retention?: {
    window_requests: number;
    latency_samples: number;
    latency_sample_capacity: number;
    latency_samples_truncated: boolean;
  };
  window_seconds: number;
  totals: { requests: number; errors: number; rejections: number };
  aggregate: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    throughput_rps: number;
    error_rate: number;
  };
  by_version: VersionMetrics[];
  timeseries: { at: string; requests: number; errors: number; p95_ms: number }[];
  queue: {
    queue_depth: number;
    inflight: number;
    batches: number;
    max_batch_size: number;
    queue_capacity_per_version: number;
  };
};
export type Chaos = { enabled: boolean; error_rate: number; latency_ms: number };
export type Frame = { label: string; policy: Policy; metrics: Metrics; events: AuditEvent[] };
export type Recording = {
  captured_at: string;
  source: string;
  models: Model[];
  samples: Sample[];
  frames: Frame[];
  benchmarks?: Record<string, unknown>;
};
export type Prediction = {
  request_id: string;
  version: string;
  prediction: number;
  probabilities: number[];
  latency_ms: number;
  queue_ms: number;
  inference_ms: number;
  batch_size: number;
  policy_revision: number;
};
export type Snapshot = Frame & {
  models: Model[];
  samples: Sample[];
  demo_mode: boolean;
  chaos: Chaos;
};
