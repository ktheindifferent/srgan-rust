# srgan-sdk

TypeScript/JavaScript SDK for the [srgan-rust](https://github.com/your-org/srgan-rust) HTTP API.

## Installation

```bash
npm install srgan-sdk
# or
yarn add srgan-sdk
```

## Quick start

```typescript
import { SrganClient } from "srgan-sdk";
import { readFileSync } from "fs";

const client = new SrganClient({
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key", // optional
});

// Check server health
const health = await client.health();
console.log(health.status);       // "ok"
console.log(health.version);      // "0.2.0"
console.log(health.modelFactor);  // 4

// Upscale an image from a file
const imageBuffer = readFileSync("input.jpg");
const result = await client.upscale(imageBuffer, {
  scaleFactor: 4,
  format: "png",
  model: "natural",
});

if (result.success && result.imageData) {
  const outputBuffer = Buffer.from(result.imageData, "base64");
  writeFileSync("output.png", outputBuffer);
  console.log(
    `Upscaled ${result.metadata.originalSize} → ${result.metadata.upscaledSize} ` +
    `in ${result.metadata.processingTimeMs}ms`
  );
}
```

## API reference

### `new SrganClient(options)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | `string` | — | Base URL of the srgan-rust server |
| `apiKey` | `string` | — | Sent as the `X-API-Key` request header |
| `timeoutMs` | `number` | `30000` | Per-request timeout in milliseconds |

---

### `client.upscale(input, options?): Promise<UpscaleResult>`

Upscale a single image synchronously.

**Parameters**

| Param | Type | Description |
|-------|------|-------------|
| `input` | `Buffer \| string` | Raw image bytes (Node.js `Buffer`) or a base64-encoded image string |
| `options.scaleFactor` | `number` | Scale multiplier (default: 4) |
| `options.format` | `"png" \| "jpeg" \| "webp"` | Output format |
| `options.quality` | `number` | JPEG quality 1–100 |
| `options.model` | `"natural" \| "anime" \| "custom"` | Model preset |

**Returns** `UpscaleResult`

```typescript
interface UpscaleResult {
  success: boolean;
  imageData: string | null;       // Base64-encoded result image
  s3Url: string | null;           // Presigned S3 URL (when S3 is configured)
  error: string | null;
  metadata: {
    originalSize: [number, number];
    upscaledSize: [number, number];
    processingTimeMs: number;
    format: string;
    modelUsed: string;
  };
}
```

---

### `client.status(jobId): Promise<JobStatus>`

Poll the status of an async upscale job.

**Returns** `JobStatus`

```typescript
interface JobStatus {
  id: string;
  status: "Pending" | "Processing" | "Completed" | "Failed";
  createdAt: number;   // Unix timestamp (seconds)
  updatedAt: number;
  resultUrl: string | null;
  resultData: string | null;  // Base64-encoded result when completed
  error: string | null;
}
```

---

### `client.health(): Promise<HealthResponse>`

Check server health and version information.

**Returns** `HealthResponse`

```typescript
interface HealthResponse {
  status: string;         // "ok" when healthy
  modelLoaded: boolean;
  model: string;
  modelFactor: number;    // e.g. 4 for 4× upscaling
  version: string;        // server version, e.g. "0.2.0"
  uptime: number;         // Unix timestamp of server start
}
```

---

### `client.batchUpscale(images, options?): Promise<BatchUpscaleResult>`

Upscale up to 10 images in a single synchronous request.

```typescript
const result = await client.batchUpscale([buf1, buf2], { format: "png" });
for (const item of result.results) {
  console.log(item.index, item.success, item.processingTimeMs);
}
```

---

### `SrganError`

All HTTP-level errors are thrown as `SrganError` instances:

```typescript
import { SrganClient, SrganError } from "srgan-sdk";

try {
  await client.upscale(buffer);
} catch (err) {
  if (err instanceof SrganError) {
    console.error(err.statusCode, err.message);
  }
}
```

## CommonJS usage

```javascript
const { SrganClient } = require("srgan-sdk");
```

## License

MIT
