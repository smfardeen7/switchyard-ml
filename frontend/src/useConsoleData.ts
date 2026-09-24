import { useCallback, useEffect, useState } from 'react';
import { read, recordedMode } from './api';
import type {
  AuditEvent,
  Chaos,
  Metrics,
  Model,
  Policy,
  Recording,
  Sample,
  Snapshot,
} from './types';

export function useConsoleData() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState('');
  const [attempt, setAttempt] = useState(0);
  const refresh = useCallback(() => setAttempt((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        if (recordedMode) {
          const response = await fetch(`${import.meta.env.BASE_URL}demo.json`, {
            signal: controller.signal,
          });
          if (!response.ok) throw new Error(`Recording could not load (${response.status}).`);
          const data = (await response.json()) as Recording;
          if (!data.frames?.length || !data.models?.length || !Array.isArray(data.samples))
            throw new Error(
              'The recording is incomplete. Generate it with the real integration scenario.',
            );
          if (!controller.signal.aborted) {
            setRecording(data);
            setError('');
            setUpdatedAt(data.captured_at);
          }
        } else {
          const [health, models, samples, policy, metrics, events, chaos] = await Promise.all([
            read<{ ready: boolean; demo_mode: boolean }>('/health', controller.signal),
            read<{ models: Model[] }>('/v1/models', controller.signal),
            read<{ samples: Sample[] }>('/v1/samples', controller.signal),
            read<Policy>('/v1/policy', controller.signal),
            read<Metrics>('/v1/metrics', controller.signal),
            read<{ events: AuditEvent[] }>('/v1/events', controller.signal),
            read<Chaos>('/v1/chaos', controller.signal),
          ]);
          if (!health.ready) throw new Error('Models are still warming. The console will retry.');
          if (!controller.signal.aborted) {
            setSnapshot({
              label: 'Live',
              models: models.models,
              samples: samples.samples,
              policy,
              metrics,
              events: events.events,
              demo_mode: health.demo_mode,
              chaos,
            });
            setUpdatedAt(new Date().toISOString());
            setError('');
          }
        }
      } catch (error) {
        if (!controller.signal.aborted)
          setError(error instanceof Error ? error.message : 'Could not connect to the API.');
      } finally {
        if (!recordedMode && !controller.signal.aborted) timer = setTimeout(load, 2000);
      }
    }
    void load();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [attempt]);

  const selected = recording?.frames[Math.min(frameIndex, recording.frames.length - 1)];
  const data: Snapshot | null =
    recordedMode && selected && recording
      ? {
          ...selected,
          models: recording.models,
          samples: recording.samples,
          demo_mode: false,
          chaos: { enabled: false, error_rate: 0, latency_ms: 0 },
        }
      : snapshot;
  return { data, recording, frameIndex, setFrameIndex, error, updatedAt, refresh };
}
