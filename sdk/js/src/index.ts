/**
 * SRGAN-Rust JavaScript / TypeScript SDK
 *
 * Provides a typed client for the SRGAN-Rust REST API.
 *
 * @example
 * ```typescript
 * import { SrganClient } from "@srgan-rust/sdk";
 * import { readFileSync } from "fs";
 *
 * const client = new SrganClient({ apiKey: "sk-..." });
 *
 * const raw = readFileSync("photo.jpg");
 * const result = await client.upscale(raw, { model: "natural" });
 * writeFileSync("photo_4x.png", Buffer.from(result.image, "base64"));
 * ```
 */

import {
  AsyncJobResponse,
  BatchRequest,
  BatchResponse,
  HealthResponse,
  JobStatusResponse,
  ModelInfo,
  SrganAuthError,
  SrganClientConfig,
  SrganError,
  SrganRateLimitError,
  SrganValidationError,
  UpscaleRequest,
  UpscaleResponse,
  ModelLabel,
} from "./types";

export * from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL = "http://localhost:8080";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_BASE_DELAY_MS = 1_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function toBase64(data: Buffer | Uint8Array | string): string {
  if (typeof data === "string") return data; // assume already base64
  return Buffer.from(data).toString("base64");
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class SrganClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly retryBaseDelayMs: number;

  constructor(config: SrganClientConfig = {}) {
    this.baseUrl = (config.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.retryBaseDelayMs = config.retryBaseDelayMs ?? DEFAULT_RETRY_BASE_DELAY_MS;
  }

  // ── Core inference ────────────────────────────────────────────────────────

  /**
   * Upscale an image synchronously.
   *
   * For images > 2 MP the server may queue the job and return 202.
   * In that case this method automatically polls until the job completes.
   *
   * @param image  Raw image bytes or a base64 string.
   * @param opts   Optional request overrides (model, quality, webhook).
   */
  async upscale(
    image: Buffer | Uint8Array | string,
    opts: Omit<UpscaleRequest, "image"> = {},
  ): Promise<UpscaleResponse> {
    const body: UpscaleRequest = { image: toBase64(image), ...opts };
    const response = await this.request<UpscaleResponse | AsyncJobResponse>(
      "POST",
      "/api/v1/upscale",
      body,
    );

    // If the server queued the job, poll for completion.
    if ("job_id" in response) {
      return this.waitForJob(response.job_id);
    }
    return response as UpscaleResponse;
  }

  /**
   * Submit an upscale job asynchronously (fire-and-forget).
   * Use {@link getJob} or {@link waitForJob} to retrieve the result.
   */
  async upscaleAsync(
    image: Buffer | Uint8Array | string,
    opts: Omit<UpscaleRequest, "image"> = {},
  ): Promise<AsyncJobResponse> {
    const body: UpscaleRequest = { image: toBase64(image), ...opts };
    return this.request<AsyncJobResponse>("POST", "/api/v1/upscale/async", body);
  }

  // ── Batch processing ──────────────────────────────────────────────────────

  /**
   * Upscale multiple images in a single call.
   *
   * For batches > 10 images the server processes them asynchronously and
   * this method polls until completion.
   */
  async batch(
    images: Array<Buffer | Uint8Array | string>,
    opts: Omit<BatchRequest, "images"> = {},
  ): Promise<BatchResponse> {
    const body: BatchRequest = {
      images: images.map(toBase64),
      ...opts,
    };
    const response = await this.request<BatchResponse>("POST", "/api/v1/batch", body);

    if (response.status === "processing") {
      return this.waitForBatch(response.batch_id);
    }
    return response;
  }

  // ── Job management ────────────────────────────────────────────────────────

  /** Retrieve the current status (and result) of a job. */
  async getJob(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>("GET", `/api/v1/job/${encodeURIComponent(jobId)}`);
  }

  /**
   * Poll a job until it reaches a terminal state.
   *
   * @param jobId         Job to wait for.
   * @param pollIntervalMs  How often to poll (default: 2 000 ms).
   * @param timeoutMs     Give up after this many ms (default: 10 minutes).
   */
  async waitForJob(
    jobId: string,
    pollIntervalMs = 2_000,
    timeoutMs = 600_000,
  ): Promise<UpscaleResponse> {
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      const status = await this.getJob(jobId);

      if (status.status === "completed") {
        if (!status.result) {
          throw new SrganError("Job completed but result is missing");
        }
        return status.result;
      }

      if (status.status === "failed") {
        throw new SrganError(`Job ${jobId} failed: ${status.error ?? "unknown error"}`);
      }

      if (status.status === "timed_out") {
        throw new SrganError(`Job ${jobId} timed out on the server`);
      }

      await sleep(pollIntervalMs);
    }

    throw new SrganError(
      `Timed out waiting for job ${jobId} after ${timeoutMs / 1000}s`,
    );
  }

  // ── Service info ──────────────────────────────────────────────────────────

  /** List the models available on this server. */
  async listModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>("GET", "/api/models");
  }

  /** Check whether the server is healthy. */
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/api/health");
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private async waitForBatch(batchId: string, pollIntervalMs = 2_000): Promise<BatchResponse> {
    while (true) {
      const status = await this.request<BatchResponse>(
        "GET",
        `/api/v1/batch/${encodeURIComponent(batchId)}`,
      );
      if (status.status !== "processing") return status;
      await sleep(pollIntervalMs);
    }
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    return headers;
  }

  private async request<T>(
    method: "GET" | "POST" | "DELETE",
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers = this.buildHeaders();

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timeoutHandle = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const response = await fetch(url, {
          method,
          headers,
          body: body !== undefined ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutHandle);

        if (response.ok) {
          return (await response.json()) as T;
        }

        // Handle specific error codes.
        if (response.status === 401) {
          throw new SrganAuthError();
        }

        if (response.status === 429) {
          const retryAfter = parseInt(response.headers.get("Retry-After") ?? "60", 10);
          if (attempt < this.maxRetries) {
            const delay = Math.min(
              retryAfter * 1000,
              this.retryBaseDelayMs * Math.pow(2, attempt),
            );
            await sleep(delay);
            lastError = new SrganRateLimitError(undefined, retryAfter);
            continue;
          }
          throw new SrganRateLimitError(undefined, retryAfter);
        }

        if (response.status === 422) {
          const text = await response.text().catch(() => "");
          throw new SrganValidationError(text || "Validation error");
        }

        const text = await response.text().catch(() => "");
        throw new SrganError(`HTTP ${response.status}: ${text}`, response.status);
      } catch (err) {
        clearTimeout(timeoutHandle);
        if (err instanceof SrganError) throw err;
        // Network errors → retry with backoff.
        lastError = err instanceof Error ? err : new Error(String(err));
        if (attempt < this.maxRetries) {
          await sleep(this.retryBaseDelayMs * Math.pow(2, attempt));
        }
      }
    }

    throw new SrganError(
      `Request to ${url} failed after ${this.maxRetries + 1} attempts: ${lastError?.message}`,
    );
  }
}

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

/**
 * Create a pre-configured {@link SrganClient} from environment variables.
 *
 * Reads:
 * - `SRGAN_API_KEY`  → apiKey
 * - `SRGAN_BASE_URL` → baseUrl
 */
export function createClientFromEnv(): SrganClient {
  return new SrganClient({
    apiKey: process.env.SRGAN_API_KEY,
    baseUrl: process.env.SRGAN_BASE_URL,
  });
}

/** Encode a file path to a base64 string (Node.js only). */
export function imageFileToBase64(filePath: string): string {
  // Dynamic require so the module still loads in browser environments.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const fs = require("fs") as typeof import("fs");
  return fs.readFileSync(filePath).toString("base64");
}
