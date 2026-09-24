# Console design

A model-serving operator needs to see where traffic goes, whether the release is safe, and what changed. The visual center is the traffic/latency panel; the right-hand release panel is the action surface. The lower half contains reproducible model metadata, an actual digit inference request and the audit trail.

Palette: deep navy #102c3f, paper #f4f6f5, white #ffffff, teal #007a78, cyan #4bd5cf, ochre #a56f15, error #b54442. System sans typography for labels, tabular numerals for metrics, monospace only for artifact IDs and request IDs. Compact left navigation; spacious workspace; controls do not look like marketing CTAs. Rounded panels distinguish sections without gradients or decorative shadows.

Recording mode must be visually explicit at all times, label capture time, and never perform POST requests. A frame selector inspects captured observations, not a simulated live animation. Measured values appear only after data arrives. Configured guardrails are distinguished from observed metrics. Live writes use current optimistic revision and show server errors without optimistic success.

All controls have labels, focus outlines and disabled explanations. Desktop uses a wide observability column and compact release column. Below 1050px panels stack; below 640px navigation becomes a compact header. Charts have equivalent values in visible labels/tables. Avoid motion except user-driven progress, respect reduced motion.
