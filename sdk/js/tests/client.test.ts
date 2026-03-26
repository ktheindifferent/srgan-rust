import {
  SrganClient,
  SrganAuthError,
  SrganRateLimitError,
  SrganValidationError,
  SrganError,
  createClientFromEnv,
} from "../src/index";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Replace the global `fetch` with a mock for the duration of a test. */
function mockFetch(handler: (url: string, init: RequestInit) => Promise<Response>): jest.Mock {
  const mock = jest.fn().mockImplementation(handler);
  (global as any).fetch = mock;
  return mock;
}

function jsonResponse(body: unknown, status = 200): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

function errorResponse(status: number, body = ""): Promise<Response> {
  return Promise.resolve(new Response(body, { status }));
}

const TINY_PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

// ---------------------------------------------------------------------------
// Constructor / config
// ---------------------------------------------------------------------------

describe("SrganClient constructor", () => {
  it("uses defaults when no config given", () => {
    const client = new SrganClient();
    expect(client).toBeDefined();
  });

  it("strips trailing slash from baseUrl", () => {
    // Access private field for testing purposes.
    const client = new SrganClient({ baseUrl: "http://localhost:9090/" }) as any;
    expect(client.baseUrl).toBe("http://localhost:9090");
  });
});

// ---------------------------------------------------------------------------
// upscale – sync path
// ---------------------------------------------------------------------------

describe("SrganClient.upscale", () => {
  afterEach(() => jest.restoreAllMocks());

  it("returns UpscaleResponse on success", async () => {
    mockFetch(() =>
      jsonResponse({
        image: TINY_PNG_B64,
        content_type: "image/png",
        width: 1,
        height: 1,
        inference_ms: 42,
      }),
    );

    const client = new SrganClient();
    const result = await client.upscale(Buffer.from(TINY_PNG_B64, "base64"));

    expect(result.image).toBe(TINY_PNG_B64);
    expect(result.content_type).toBe("image/png");
    expect(result.inference_ms).toBe(42);
  });

  it("polls when server returns async job response", async () => {
    let callCount = 0;
    mockFetch((_url, _init) => {
      callCount++;
      if (callCount === 1) {
        // First call: server queued the job
        return jsonResponse({ job_id: "job-1", status: "pending", created_at: "" }, 202);
      }
      // Subsequent calls: job status polling
      if (callCount === 2) {
        return jsonResponse({ job_id: "job-1", status: "processing", created_at: "" });
      }
      return jsonResponse({
        job_id: "job-1",
        status: "completed",
        created_at: "",
        result: {
          image: TINY_PNG_B64,
          content_type: "image/png",
          width: 4,
          height: 4,
          inference_ms: 100,
        },
      });
    });

    const client = new SrganClient({ retryBaseDelayMs: 1 });
    const result = await client.upscale(TINY_PNG_B64, {}, /* pollIntervalMs */ );
    expect(result.width).toBe(4);
    expect(callCount).toBe(3);
  });

  it("throws SrganAuthError on 401", async () => {
    mockFetch(() => errorResponse(401));

    const client = new SrganClient({ maxRetries: 0 });
    await expect(client.upscale(TINY_PNG_B64)).rejects.toBeInstanceOf(SrganAuthError);
  });

  it("throws SrganRateLimitError on 429", async () => {
    mockFetch(() =>
      Promise.resolve(
        new Response("", {
          status: 429,
          headers: { "Retry-After": "30" },
        }),
      ),
    );

    const client = new SrganClient({ maxRetries: 0 });
    await expect(client.upscale(TINY_PNG_B64)).rejects.toBeInstanceOf(SrganRateLimitError);
  });

  it("throws SrganValidationError on 422", async () => {
    mockFetch(() => errorResponse(422, "image too large"));

    const client = new SrganClient({ maxRetries: 0 });
    await expect(client.upscale(TINY_PNG_B64)).rejects.toBeInstanceOf(SrganValidationError);
  });

  it("throws SrganError on non-OK status", async () => {
    mockFetch(() => errorResponse(500, "internal error"));

    const client = new SrganClient({ maxRetries: 0 });
    await expect(client.upscale(TINY_PNG_B64)).rejects.toBeInstanceOf(SrganError);
  });
});

// ---------------------------------------------------------------------------
// waitForJob
// ---------------------------------------------------------------------------

describe("SrganClient.waitForJob", () => {
  afterEach(() => jest.restoreAllMocks());

  it("resolves when job completes", async () => {
    mockFetch(() =>
      jsonResponse({
        job_id: "j1",
        status: "completed",
        created_at: "",
        result: { image: TINY_PNG_B64, content_type: "image/png", width: 1, height: 1, inference_ms: 5 },
      }),
    );

    const client = new SrganClient();
    const result = await client.waitForJob("j1");
    expect(result.image).toBe(TINY_PNG_B64);
  });

  it("rejects when job fails", async () => {
    mockFetch(() =>
      jsonResponse({ job_id: "j1", status: "failed", created_at: "", error: "OOM" }),
    );

    const client = new SrganClient();
    await expect(client.waitForJob("j1")).rejects.toThrow("OOM");
  });

  it("rejects when job times out on server", async () => {
    mockFetch(() =>
      jsonResponse({ job_id: "j1", status: "timed_out", created_at: "" }),
    );

    const client = new SrganClient();
    await expect(client.waitForJob("j1")).rejects.toThrow("timed out on the server");
  });

  it("rejects when client polling timeout exceeded", async () => {
    mockFetch(() =>
      jsonResponse({ job_id: "j1", status: "processing", created_at: "" }),
    );

    const client = new SrganClient();
    await expect(client.waitForJob("j1", 1, 10)).rejects.toThrow("Timed out waiting");
  });
});

// ---------------------------------------------------------------------------
// health / listModels
// ---------------------------------------------------------------------------

describe("SrganClient.health", () => {
  it("returns health status", async () => {
    mockFetch(() =>
      jsonResponse({
        status: "ok",
        version: "0.2.0",
        uptime_seconds: 1234,
        model_loaded: true,
        images_processed: 99,
      }),
    );

    const client = new SrganClient();
    const h = await client.health();
    expect(h.status).toBe("ok");
    expect(h.model_loaded).toBe(true);
  });
});

describe("SrganClient.listModels", () => {
  it("returns model list", async () => {
    mockFetch(() =>
      jsonResponse([
        { label: "natural", description: "Natural images", scale_factor: 4 },
        { label: "anime", description: "Anime images", scale_factor: 4 },
      ]),
    );

    const client = new SrganClient();
    const models = await client.listModels();
    expect(models).toHaveLength(2);
    expect(models[0].label).toBe("natural");
  });
});

// ---------------------------------------------------------------------------
// createClientFromEnv
// ---------------------------------------------------------------------------

describe("createClientFromEnv", () => {
  it("reads SRGAN_API_KEY from environment", () => {
    process.env.SRGAN_API_KEY = "sk-test-key";
    process.env.SRGAN_BASE_URL = "http://test.local";
    const client = createClientFromEnv() as any;
    expect(client.apiKey).toBe("sk-test-key");
    expect(client.baseUrl).toBe("http://test.local");
    delete process.env.SRGAN_API_KEY;
    delete process.env.SRGAN_BASE_URL;
  });
});

// ---------------------------------------------------------------------------
// Error hierarchy
// ---------------------------------------------------------------------------

describe("error types", () => {
  it("SrganAuthError is a SrganError", () => {
    expect(new SrganAuthError()).toBeInstanceOf(SrganError);
    expect(new SrganAuthError().statusCode).toBe(401);
  });

  it("SrganRateLimitError carries retryAfterSeconds", () => {
    const err = new SrganRateLimitError("Too many requests", 60);
    expect(err.retryAfterSeconds).toBe(60);
    expect(err.statusCode).toBe(429);
  });

  it("SrganValidationError has status 422", () => {
    expect(new SrganValidationError("bad input").statusCode).toBe(422);
  });
});
