# @srgan-rust/sdk

JavaScript / TypeScript SDK for the [SRGAN-Rust](https://github.com/ktheindifferent/srgan-rust)
image upscaling API.

## Installation

```bash
npm install @srgan-rust/sdk
# or
yarn add @srgan-rust/sdk
```

## Quick start

```typescript
import { SrganClient } from "@srgan-rust/sdk";
import { readFileSync, writeFileSync } from "fs";

const client = new SrganClient({
  baseUrl: "http://localhost:8080",
  apiKey: "sk-...",           // omit for unauthenticated local usage
});

// Upscale a file and save the result
const raw = readFileSync("photo.jpg");
const result = await client.upscale(raw, { model: "natural" });
writeFileSync("photo_4x.png", Buffer.from(result.image, "base64"));

console.log(`Done in ${result.inference_ms}ms`);
```

## Configuration

```typescript
const client = new SrganClient({
  baseUrl: "http://localhost:8080",  // default
  apiKey: "sk-...",                   // required for hosted servers
  timeoutMs: 30_000,                  // per-request timeout (default 30 s)
  maxRetries: 3,                       // retries on 429 / network error
  retryBaseDelayMs: 1_000,             // exponential backoff base
});
```

You can also build a client from environment variables:

```bash
export SRGAN_API_KEY=sk-...
export SRGAN_BASE_URL=https://your-server.example.com
```

```typescript
import { createClientFromEnv } from "@srgan-rust/sdk";
const client = createClientFromEnv();
```

## API reference

### `client.upscale(image, options?)`

Upscale an image synchronously.  For images > 2 MP the server will process
the job asynchronously; the SDK polls automatically until the job completes.

```typescript
const result = await client.upscale(imageBytes, {
  model: "anime",         // "natural" | "anime" | "bilinear"  (default "natural")
  jpeg_quality: 90,       // 1–100, only used when the output is JPEG
  webhook_url: "https://…", // called by the server when the job completes
});
// result: { image: string (base64), content_type, width, height, inference_ms }
```

### `client.upscaleAsync(image, options?)`

Submit a job without waiting for the result.

```typescript
const job = await client.upscaleAsync(imageBytes);
console.log(job.job_id); // use with getJob / waitForJob
```

### `client.waitForJob(jobId, pollIntervalMs?, timeoutMs?)`

Poll until a job reaches a terminal state.

```typescript
const result = await client.waitForJob(job.job_id, 2_000, 600_000);
```

### `client.batch(images, options?)`

Upscale multiple images.  For batches > 10 the SDK polls for completion.

```typescript
const result = await client.batch([img1, img2, img3], { model: "natural" });
console.log(`${result.completed}/${result.total} images done`);
```

### `client.health()`

```typescript
const { status, version, uptime_seconds, model_loaded } = await client.health();
```

### `client.listModels()`

```typescript
const models = await client.listModels();
// [{ label, description, scale_factor }, ...]
```

## Error handling

All errors extend `SrganError`.

| Class | Status | Meaning |
|-------|--------|---------|
| `SrganAuthError` | 401 | Invalid or missing API key |
| `SrganRateLimitError` | 429 | Quota exceeded; check `.retryAfterSeconds` |
| `SrganValidationError` | 422 | Input validation failed |
| `SrganError` | other | Generic API or network error |

```typescript
import { SrganRateLimitError } from "@srgan-rust/sdk";

try {
  await client.upscale(imageBytes);
} catch (err) {
  if (err instanceof SrganRateLimitError) {
    console.log(`Retry after ${err.retryAfterSeconds}s`);
  } else {
    throw err;
  }
}
```

## Development

```bash
npm install
npm run build   # compile TypeScript → dist/
npm test        # run Jest test suite
npm run test:coverage
```

Dry-run the publish to verify the package is ready:

```bash
npm publish --dry-run
```

## Models

| Label | Best for |
|-------|----------|
| `natural` | Photographs, landscapes, portraits |
| `anime` | Animation, manga, flat-colour art |
| `bilinear` | Fast preview / baseline |

All models upscale 4× (e.g. 512×512 → 2048×2048).

## License

MIT
