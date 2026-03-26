// ── Types ─────────────────────────────────────────────────────────────────────

export interface UpscaleOptions {
  /** Scale factor (default: 4) */
  scaleFactor?: number;
  /** Output image format: "png" | "jpeg" | "webp" */
  format?: "png" | "jpeg" | "webp";
  /** JPEG quality 1–100 (only relevant when format is "jpeg") */
  quality?: number;
  /** Model preset: "natural" | "anime" | "custom" */
  model?: "natural" | "anime" | "custom";
}

export interface ResponseMetadata {
  originalSize: [number, number];
  upscaledSize: [number, number];
  processingTimeMs: number;
  format: string;
  modelUsed: string;
}

export interface UpscaleResult {
  success: boolean;
  /** Base64-encoded result image (present when S3 is not configured) */
  imageData: string | null;
  /** Presigned S3 URL (present when server has S3 configured) */
  s3Url: string | null;
  error: string | null;
  metadata: ResponseMetadata;
}

export type JobStatusKind = "Pending" | "Processing" | "Completed" | "Failed";

export interface JobStatus {
  id: string;
  status: JobStatusKind;
  createdAt: number;
  updatedAt: number;
  resultUrl: string | null;
  /** Base64-encoded result image when job is completed */
  resultData: string | null;
  error: string | null;
}

export interface HealthResponse {
  status: "ok" | string;
  modelLoaded: boolean;
  model: string;
  modelFactor: number;
  version: string;
  /** Unix timestamp (seconds) */
  uptime: number;
}

export interface BatchImageResult {
  index: number;
  success: boolean;
  imageData: string | null;
  error: string | null;
  processingTimeMs: number;
}

export interface BatchUpscaleResult {
  success: boolean;
  results: BatchImageResult[];
  total: number;
  successful: number;
  failed: number;
  totalProcessingTimeMs: number;
}

export interface SrganClientOptions {
  /** Base URL of the srgan-rust server, e.g. "http://localhost:8080" */
  baseUrl: string;
  /** Optional API key sent as the `X-API-Key` header */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 30 000) */
  timeoutMs?: number;
}

// ── Internal helpers ───────────────────────────────────────────────────────────

function toBase64(input: Buffer | string): string {
  if (typeof input === "string") {
    // Assume it's already base64 if it doesn't look like a file path / data URL.
    // Callers who pass raw base64 strings will get them forwarded as-is.
    return input;
  }
  // Node.js Buffer
  return input.toString("base64");
}

/** Convert snake_case server keys to camelCase for the UpscaleResult */
function mapUpscaleResponse(raw: Record<string, unknown>): UpscaleResult {
  const meta = (raw.metadata ?? {}) as Record<string, unknown>;
  return {
    success: Boolean(raw.success),
    imageData: (raw.image_data as string | null) ?? null,
    s3Url: (raw.s3_url as string | null) ?? null,
    error: (raw.error as string | null) ?? null,
    metadata: {
      originalSize: (meta.original_size as [number, number]) ?? [0, 0],
      upscaledSize: (meta.upscaled_size as [number, number]) ?? [0, 0],
      processingTimeMs: (meta.processing_time_ms as number) ?? 0,
      format: (meta.format as string) ?? "",
      modelUsed: (meta.model_used as string) ?? "",
    },
  };
}

function mapJobStatus(raw: Record<string, unknown>): JobStatus {
  return {
    id: (raw.id as string) ?? "",
    status: (raw.status as JobStatusKind) ?? "Pending",
    createdAt: (raw.created_at as number) ?? 0,
    updatedAt: (raw.updated_at as number) ?? 0,
    resultUrl: (raw.result_url as string | null) ?? null,
    resultData: (raw.result_data as string | null) ?? null,
    error: (raw.error as string | null) ?? null,
  };
}

function mapHealthResponse(raw: Record<string, unknown>): HealthResponse {
  return {
    status: (raw.status as string) ?? "unknown",
    modelLoaded: Boolean(raw.model_loaded),
    model: (raw.model as string) ?? "",
    modelFactor: (raw.model_factor as number) ?? 0,
    version: (raw.version as string) ?? "",
    uptime: (raw.uptime as number) ?? 0,
  };
}

function mapBatchResponse(raw: Record<string, unknown>): BatchUpscaleResult {
  const results = ((raw.results as unknown[]) ?? []).map((r) => {
    const item = r as Record<string, unknown>;
    return {
      index: (item.index as number) ?? 0,
      success: Boolean(item.success),
      imageData: (item.image_data as string | null) ?? null,
      error: (item.error as string | null) ?? null,
      processingTimeMs: (item.processing_time_ms as number) ?? 0,
    };
  });
  return {
    success: Boolean(raw.success),
    results,
    total: (raw.total as number) ?? 0,
    successful: (raw.successful as number) ?? 0,
    failed: (raw.failed as number) ?? 0,
    totalProcessingTimeMs: (raw.total_processing_time_ms as number) ?? 0,
  };
}

// ── SrganClient ────────────────────────────────────────────────────────────────

export class SrganClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;

  constructor(options: SrganClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.headers = {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(options.apiKey ? { "X-API-Key": options.apiKey } : {}),
    };
  }

  // ── Private fetch wrapper ────────────────────────────────────────────────────

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const json = (await response.json()) as { error?: string };
        if (json.error) message = json.error;
      } catch {
        // ignore parse errors
      }
      throw new SrganError(message, response.status);
    }

    return response.json() as Promise<T>;
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Upscale an image synchronously.
   *
   * @param input  A Node.js `Buffer` containing raw image bytes, or a
   *               base64-encoded image string.
   * @param options  Optional upscaling parameters.
   * @returns  The upscaled image data and metadata.
   */
  async upscale(
    input: Buffer | string,
    options?: UpscaleOptions
  ): Promise<UpscaleResult> {
    const payload: Record<string, unknown> = {
      image_data: toBase64(input),
    };
    if (options?.scaleFactor !== undefined)
      payload.scale_factor = options.scaleFactor;
    if (options?.format !== undefined) payload.format = options.format;
    if (options?.quality !== undefined) payload.quality = options.quality;
    if (options?.model !== undefined) payload.model = options.model;

    const raw = await this.request<Record<string, unknown>>(
      "POST",
      "/api/v1/upscale",
      payload
    );
    return mapUpscaleResponse(raw);
  }

  /**
   * Poll the status of an async upscale job.
   *
   * @param jobId  The job ID returned by an async upscale request.
   */
  async status(jobId: string): Promise<JobStatus> {
    const raw = await this.request<Record<string, unknown>>(
      "GET",
      `/api/v1/job/${encodeURIComponent(jobId)}`
    );
    return mapJobStatus(raw);
  }

  /**
   * Check server health.
   */
  async health(): Promise<HealthResponse> {
    const raw = await this.request<Record<string, unknown>>(
      "GET",
      "/api/v1/health"
    );
    return mapHealthResponse(raw);
  }

  /**
   * Upscale a batch of images synchronously (≤10 images).
   *
   * @param images  Array of `Buffer` or base64 strings.
   * @param options  Optional format / model overrides (scale_factor not
   *                 supported in batch mode).
   */
  async batchUpscale(
    images: Array<Buffer | string>,
    options?: Pick<UpscaleOptions, "format" | "model">
  ): Promise<BatchUpscaleResult> {
    const payload: Record<string, unknown> = {
      images: images.map(toBase64),
    };
    if (options?.format !== undefined) payload.format = options.format;
    if (options?.model !== undefined) payload.model = options.model;

    const raw = await this.request<Record<string, unknown>>(
      "POST",
      "/api/v1/batch",
      payload
    );
    return mapBatchResponse(raw);
  }
}

// ── Error class ────────────────────────────────────────────────────────────────

export class SrganError extends Error {
  readonly statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.name = "SrganError";
    this.statusCode = statusCode;
  }
}
