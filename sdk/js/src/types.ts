/** Model types supported by the SRGAN-Rust API. */
export type ModelLabel = "natural" | "anime" | "bilinear";

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
