export const recordedMode = import.meta.env.VITE_DEMO_MODE === '1';
export const apiRoot = (import.meta.env.VITE_API_ROOT || '').replace(/\/$/, '');
export async function read<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${apiRoot}${path}`, { signal });
  return parse<T>(response);
}
async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === 'string' ? body.detail : `Request failed (${response.status}).`,
    );
  }
  return response.json();
}
export async function write<T>(
  path: string,
  body: unknown,
  token: string,
  signal?: AbortSignal,
): Promise<T> {
  if (recordedMode)
    throw new Error('Recorded mode is read-only. Start the local API to use live controls.');
  if (!token.trim()) throw new Error('Enter your API token to send a request.');
  return parse<T>(
    await fetch(`${apiRoot}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
      signal,
    }),
  );
}
