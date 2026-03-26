/**
 * Model types supported by the SRGAN-Rust API.
 *
 * - `"natural"` — General-purpose SRGAN model for photographic content (×4).
 * - `"anime"` — SRGAN model trained on anime/illustration datasets (×4).
 * - `"waifu2x"` — Waifu2x-compatible anime/illustration upscaler with
 *   configurable noise reduction (levels 0–3) and scale (×1 denoise-only or
 *   ×2 upscale). Uses a VGG-style CNN architecture optimised for line art,
 *   flat colour regions, and JPEG artifact removal common in anime imagery.
 *   Pass `waifu2x_noise_level` and `waifu2x_scale` in the request for
 *   fine-grained control, or use bare `"waifu2x"` for defaults (noise=1, ×2).
 * - `"real-esrgan"` — Real-ESRGAN model for photorealistic content (×4).
 * - `"real-esrgan-anime"` — Real-ESRGAN variant tuned for anime (×4).
 * - `"real-esrgan-x2"` — Real-ESRGAN at ×2 scale.
 * - `"bilinear"` — Fast bilinear interpolation (no neural network).
 */
export type ModelLabel =
  | "natural"
  | "anime"
  | "waifu2x"
  | "real-esrgan"
  | "real-esrgan-anime"
  | "real-esrgan-x2"
  | "bilinear";

// ---------------------------------------------------------------------------
// SSE streaming types
// ---------------------------------------------------------------------------

/** Union of all event shapes emitted by the SSE stream endpoint. */
export type SSEProgressEvent =
  | SSEQueuedEvent
  | SSEStartedEvent
  | SSEProgressEventData
  | SSEDoneEvent
  | SSEErrorEvent;

export interface SSEQueuedEvent {
  event: "queued";
  job_id: string;
  /** Number of jobs ahead of this one in the queue. */
  position: number;
}

export interface SSEStartedEvent {
  event: "started";
  job_id: string;
}

export interface SSEProgressEventData {
  event: "progress";
  job_id: string;
  /** Completion percentage 0–100. */
  percent: number;
}

export interface SSEDoneEvent {
  event: "done";
  job_id: string;
  /** URL to download the upscaled result. */
  output_url: string;
}

export interface SSEErrorEvent {
  event: "error";
  job_id: string;
  message: string;
}

/** Callback signature for {@link SrganClient.upscaleStream}. */
export type ProgressCallback = (event: SSEProgressEvent) => void;

/** Tier-based API access levels. */
export type Tier = "free" | "pro" | "enterprise";

/** Job execution status. */
export type JobStatus = "pending" | "processing" | "completed" | "failed" | "timed_out";

// ---------------------------------------------------------------------------
// Request / response shapes
// ---------------------------------------------------------------------------

export interface UpscaleRequest {
  /** Base64-encoded image bytes. */
  image: string;
  /** Model to use (default: "natural"). */
  model?: ModelLabel;
  /** Waifu2x noise-reduction level 0–3 (default: 1). Only used when model is "waifu2x". */
  waifu2x_noise_level?: 0 | 1 | 2 | 3;
  /** Waifu2x scale factor: 1 (denoise-only) or 2 (upscale). Default: 2. Only used when model is "waifu2x". */
  waifu2x_scale?: 1 | 2;
  /** Waifu2x content style hint: "anime" (default), "photo", or "artwork". Only used when model is "waifu2x". */
  waifu2x_style?: "anime" | "photo" | "artwork";
  /** JPEG quality 1–100 (only used when the output is JPEG). */
  jpeg_quality?: number;
  /** Optional URL to POST the result to when the job finishes. */
  webhook_url?: string;
}

export interface UpscaleResponse {
  /** Base64-encoded upscaled image bytes. */
  image: string;
  /** MIME type of the output image (e.g. "image/png"). */
  content_type: string;
  /** Original width in pixels. */
  width: number;
  /** Original height in pixels. */
  height: number;
  /** Inference duration in milliseconds. */
  inference_ms: number;
}

export interface AsyncJobResponse {
  /** Unique job identifier. */
  job_id: string;
  status: JobStatus;
  /** ISO-8601 creation timestamp. */
  created_at: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  created_at: string;
  completed_at?: string;
  /** Populated when status is "completed". */
  result?: UpscaleResponse;
  /** Populated when status is "failed". */
  error?: string;
}

export interface BatchRequest {
  /** Array of base64-encoded images. */
  images: string[];
  model?: ModelLabel;
  jpeg_quality?: number;
}

export interface BatchResponse {
  batch_id: string;
  status: "processing" | "completed" | "failed";
  completed: number;
  total: number;
  results?: UpscaleResponse[];
}

export interface ModelInfo {
  label: ModelLabel;
  description: string;
  scale_factor: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  uptime_seconds: number;
  model_loaded: boolean;
  images_processed: number;
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

export class SrganError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
  ) {
    super(message);
    this.name = "SrganError";
  }
}

export class SrganAuthError extends SrganError {
  constructor(message = "Invalid or missing API key") {
    super(message, 401);
    this.name = "SrganAuthError";
  }
}

export class SrganRateLimitError extends SrganError {
  constructor(
    message = "Rate limit exceeded",
    public readonly retryAfterSeconds?: number,
  ) {
    super(message, 429);
    this.name = "SrganRateLimitError";
  }
}

export class SrganValidationError extends SrganError {
  constructor(message: string) {
    super(message, 422);
    this.name = "SrganValidationError";
  }
}

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

export interface SrganClientConfig {
  /** API base URL (default: "http://localhost:8080"). */
  baseUrl?: string;
  /** API key (required for non-local servers). */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 30_000). */
  timeoutMs?: number;
  /** Maximum number of automatic retries on 429 (default: 3). */
  maxRetries?: number;
  /** Base backoff delay in milliseconds (default: 1_000). */
  retryBaseDelayMs?: number;
}
