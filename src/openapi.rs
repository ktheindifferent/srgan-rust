//! Swagger UI page and OpenAPI JSON endpoint.
//!
//! - `GET /docs`          — Swagger UI HTML (loads spec from /openapi.json)
//! - `GET /openapi.json`  — OpenAPI 3.0 specification

/// Returns a self-contained Swagger UI HTML page that fetches the spec from
/// `/openapi.json`.
pub fn render_swagger_ui() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRGAN API — Swagger UI</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css">
<style>
body { margin: 0; background: #fafafa; }
#topbar { background: #1b1b1b; padding: 12px 24px; display: flex; align-items: center; }
#topbar a { color: #fff; text-decoration: none; font-family: sans-serif; font-weight: 700; font-size: 1.1rem; }
#topbar .links { margin-left: auto; }
#topbar .links a { font-size: .85rem; font-weight: 400; margin-left: 16px; color: #aaa; }
#topbar .links a:hover { color: #fff; }
</style>
</head>
<body>
<div id="topbar">
  <a href="/">SRGAN API</a>
  <div class="links">
    <a href="/docs">API Docs</a>
    <a href="/openapi.json">OpenAPI Spec</a>
    <a href="/pricing">Pricing</a>
  </div>
</div>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({
  url: '/openapi.json',
  dom_id: '#swagger-ui',
  presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
  layout: 'BaseLayout',
  deepLinking: true,
  defaultModelsExpandDepth: 1
});
</script>
</body>
</html>"##
        .to_string()
}

/// Returns the full OpenAPI 3.0 JSON specification including all endpoints:
/// upscale, batch, ensemble, compare, models, organizations, analytics,
/// webhooks, auth, and system.
pub fn render_openapi_json() -> String {
    serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "SRGAN Image Super-Resolution API",
            "description": "AI-powered image and video upscaling service built in Rust. Supports single-image upscaling, batch processing, ensemble inference, INT8 quantization, and tiled processing for large images.",
            "version": "1.2.0",
            "contact": { "email": "support@srgan.dev" },
            "license": { "name": "MIT" }
        },
        "servers": [
            { "url": "https://api.srgan.dev", "description": "Production" },
            { "url": "http://localhost:8080", "description": "Local development" }
        ],
        "security": [{ "BearerAuth": [] }],
        "tags": [
            { "name": "Image", "description": "Image upscaling endpoints" },
            { "name": "Ensemble", "description": "Multi-model ensemble upscaling" },
            { "name": "Video", "description": "Video upscaling endpoints" },
            { "name": "Batch", "description": "Batch processing endpoints" },
            { "name": "Jobs", "description": "Job status and result retrieval" },
            { "name": "Models", "description": "Model listing and versioning" },
            { "name": "Compare", "description": "Side-by-side model comparison" },
            { "name": "Organizations", "description": "Organization and team management" },
            { "name": "Analytics", "description": "Usage analytics and reporting" },
            { "name": "Webhooks", "description": "Webhook management" },
            { "name": "Auth", "description": "Authentication and registration" },
            { "name": "Admin", "description": "Administrative endpoints" },
            { "name": "System", "description": "System health and status" }
        ],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API key"
                }
            },
            "schemas": {
                "UpscaleResponse": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string", "enum": ["complete", "processing", "failed"] },
                        "output_url": { "type": "string" },
                        "width": { "type": "integer" },
                        "height": { "type": "integer" },
                        "processing_time_ms": { "type": "integer" }
                    }
                },
                "JobStatus": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string", "enum": ["pending", "processing", "completed", "failed", "timed_out"] },
                        "progress": { "type": "integer", "minimum": 0, "maximum": 100 },
                        "created_at": { "type": "string", "format": "date-time" },
                        "started_at": { "type": "string", "format": "date-time", "nullable": true },
                        "completed_at": { "type": "string", "format": "date-time", "nullable": true }
                    }
                },
                "BatchRequest": {
                    "type": "object",
                    "required": ["images"],
                    "properties": {
                        "images": {
                            "type": "array",
                            "maxItems": 20,
                            "items": {
                                "type": "object",
                                "required": ["url"],
                                "properties": {
                                    "url": { "type": "string", "format": "uri" },
                                    "model": { "type": "string" }
                                }
                            }
                        },
                        "model": { "type": "string", "default": "natural" },
                        "scale": { "type": "integer", "default": 4 },
                        "format": { "type": "string", "enum": ["png", "jpeg"], "default": "png" },
                        "webhook_url": { "type": "string", "format": "uri" },
                        "webhook_secret": { "type": "string" }
                    }
                },
                "BatchResponse": {
                    "type": "object",
                    "properties": {
                        "batch_id": { "type": "string" },
                        "status": { "type": "string", "enum": ["processing", "completed", "partially_completed", "failed", "cancelled"] },
                        "total": { "type": "integer" },
                        "completed": { "type": "integer" },
                        "failed": { "type": "integer" },
                        "progress_pct": { "type": "integer", "minimum": 0, "maximum": 100 },
                        "poll_url": { "type": "string" }
                    }
                },
                "EnsembleRequest": {
                    "type": "object",
                    "required": ["image_data"],
                    "properties": {
                        "image_data": { "type": "string", "format": "byte", "description": "Base64-encoded input image" },
                        "models": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 5,
                            "items": { "type": "string", "enum": ["natural", "anime", "waifu2x", "real-esrgan"] },
                            "default": ["natural", "anime"],
                            "description": "Models to run in ensemble (2-5)"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["average", "best_ssim", "weighted_ssim"],
                            "default": "average",
                            "description": "How to combine model outputs"
                        },
                        "format": { "type": "string", "enum": ["png", "jpeg"] }
                    }
                },
                "EnsembleResponse": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string" },
                        "models_used": { "type": "array", "items": { "type": "string" } },
                        "strategy": { "type": "string" },
                        "model_scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model": { "type": "string" },
                                    "ssim": { "type": "number", "format": "float" },
                                    "processing_time_ms": { "type": "integer" }
                                }
                            }
                        },
                        "processing_time_ms": { "type": "integer" }
                    }
                },
                "VideoUpscaleResponse": {
                    "type": "object",
                    "properties": {
                        "job_id": { "type": "string" },
                        "status": { "type": "string" },
                        "frames_total": { "type": "integer" },
                        "output_url": { "type": "string" }
                    }
                },
                "ModelVersion": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "example": "srgan-v1" },
                        "name": { "type": "string", "example": "SRGAN" },
                        "version": { "type": "string", "example": "1.0.0" },
                        "upscale_factor": { "type": "integer", "example": 4 },
                        "supported_types": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["image", "video"] }
                        },
                        "created_at": { "type": "string", "format": "date-time" }
                    }
                },
                "CreateModelVersionRequest": {
                    "type": "object",
                    "required": ["id", "name", "version", "upscale_factor", "supported_types"],
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "version": { "type": "string" },
                        "upscale_factor": { "type": "integer" },
                        "supported_types": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["image", "video"] }
                        }
                    }
                },
                "ModelInfo": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "description": { "type": "string" },
                        "max_scale": { "type": "integer" },
                        "supports_quantization": { "type": "boolean" },
                        "supports_ensemble": { "type": "boolean" }
                    }
                },
                "CompareRequest": {
                    "type": "object",
                    "required": ["image_data", "models"],
                    "properties": {
                        "image_data": { "type": "string", "format": "byte", "description": "Base64-encoded input image" },
                        "models": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 4,
                            "items": { "type": "string" }
                        },
                        "metric": { "type": "string", "enum": ["psnr", "ssim", "both"], "default": "both" }
                    }
                },
                "CompareResponse": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model": { "type": "string" },
                                    "psnr": { "type": "number", "format": "float", "nullable": true },
                                    "ssim": { "type": "number", "format": "float", "nullable": true },
                                    "output_url": { "type": "string" },
                                    "processing_time_ms": { "type": "integer" }
                                }
                            }
                        },
                        "best_model": { "type": "string" }
                    }
                },
                "Organization": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "owner_api_key": { "type": "string" },
                        "members": { "type": "array", "items": { "type": "string" } },
                        "created_at": { "type": "integer" }
                    }
                },
                "CreateOrgRequest": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": { "type": "string", "minLength": 1, "maxLength": 100 }
                    }
                },
                "AddMemberRequest": {
                    "type": "object",
                    "required": ["api_key"],
                    "properties": {
                        "api_key": { "type": "string" }
                    }
                },
                "OrgUsageResponse": {
                    "type": "object",
                    "properties": {
                        "org_id": { "type": "string" },
                        "total_images": { "type": "integer" },
                        "total_pixels": { "type": "integer" },
                        "members": { "type": "integer" },
                        "period": { "type": "string" }
                    }
                },
                "AnalyticsUsageResponse": {
                    "type": "object",
                    "properties": {
                        "api_key": { "type": "string" },
                        "period": { "type": "string" },
                        "daily": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": { "type": "string", "format": "date" },
                                    "images_processed": { "type": "integer" },
                                    "pixels_processed": { "type": "integer" },
                                    "avg_latency_ms": { "type": "number", "format": "float" },
                                    "models": { "type": "object", "additionalProperties": { "type": "integer" } }
                                }
                            }
                        },
                        "totals": {
                            "type": "object",
                            "properties": {
                                "images": { "type": "integer" },
                                "pixels": { "type": "integer" },
                                "avg_latency_ms": { "type": "number" }
                            }
                        }
                    }
                },
                "AdminAnalyticsResponse": {
                    "type": "object",
                    "properties": {
                        "total_users": { "type": "integer" },
                        "total_images": { "type": "integer" },
                        "total_pixels": { "type": "integer" },
                        "active_jobs": { "type": "integer" },
                        "model_breakdown": { "type": "object", "additionalProperties": { "type": "integer" } },
                        "tier_breakdown": { "type": "object", "additionalProperties": { "type": "integer" } }
                    }
                },
                "BatchInferenceStats": {
                    "type": "object",
                    "properties": {
                        "total_batches_processed": { "type": "integer" },
                        "total_images_processed": { "type": "integer" },
                        "avg_batch_size": { "type": "number", "format": "float" },
                        "avg_batch_latency_ms": { "type": "number", "format": "float" }
                    }
                },
                "WebhookRegister": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": { "type": "string", "format": "uri" },
                        "secret": { "type": "string" },
                        "events": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["job.completed", "job.failed", "batch.completed"] }
                        }
                    }
                },
                "WebhookPayload": {
                    "type": "object",
                    "properties": {
                        "event": { "type": "string" },
                        "timestamp": { "type": "integer" },
                        "data": { "type": "object" }
                    }
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": { "type": "string" },
                        "status": { "type": "integer" }
                    }
                }
            }
        },
        "paths": {
            // ── Image upscaling ─────────────────────────────────────────────
            "/api/v1/upscale": {
                "post": {
                    "summary": "Upscale an image",
                    "operationId": "upscaleImage",
                    "tags": ["Image"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image"],
                                    "properties": {
                                        "image": { "type": "string", "format": "binary", "description": "Image file (JPEG, PNG, WebP)" },
                                        "scale": { "type": "integer", "enum": [2, 4], "default": 4 },
                                        "model": { "type": "string", "enum": ["natural", "anime", "waifu2x", "real-esrgan"], "default": "natural" },
                                        "quantize": { "type": "boolean", "default": false, "description": "Use INT8 quantized inference for ~4x faster processing" },
                                        "tile_size": { "type": "integer", "default": 0, "description": "Tile size for large images (0 = auto)" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Upscale complete", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UpscaleResponse" } } } },
                        "400": { "description": "Invalid request", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                        "401": { "description": "Unauthorized" },
                        "413": { "description": "File too large" },
                        "429": { "description": "Rate limited" }
                    }
                }
            },
            "/api/v1/upscale/async": {
                "post": {
                    "summary": "Upscale an image asynchronously",
                    "operationId": "upscaleImageAsync",
                    "tags": ["Image"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image"],
                                    "properties": {
                                        "image": { "type": "string", "format": "binary" },
                                        "scale": { "type": "integer", "enum": [2, 4], "default": 4 },
                                        "model": { "type": "string", "default": "natural" },
                                        "quantize": { "type": "boolean", "default": false },
                                        "webhook_url": { "type": "string", "format": "uri" },
                                        "webhook_secret": { "type": "string" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Job queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobStatus" } } } }
                    }
                }
            },

            // ── Ensemble ────────────────────────────────────────────────────
            "/api/v1/upscale/ensemble": {
                "post": {
                    "summary": "Ensemble upscale — run image through multiple models and combine results",
                    "description": "Runs the input image through 2-5 models (e.g. SRGAN + Real-ESRGAN + waifu2x), then combines outputs using the specified strategy. Slower but produces higher-quality results.",
                    "operationId": "ensembleUpscale",
                    "tags": ["Ensemble"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/EnsembleRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Ensemble upscale complete",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/EnsembleResponse" } } }
                        },
                        "400": { "description": "Invalid request (e.g. fewer than 2 models)", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                        "401": { "description": "Unauthorized" },
                        "429": { "description": "Rate limited" }
                    }
                }
            },

            // ── Batch ───────────────────────────────────────────────────────
            "/api/v1/batch": {
                "post": {
                    "summary": "Batch upscale multiple images",
                    "operationId": "batchUpscale",
                    "tags": ["Batch"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/BatchRequest" }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Batch queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/BatchResponse" } } } },
                        "400": { "description": "Batch too large or invalid" }
                    }
                }
            },
            "/api/v1/batch/{id}": {
                "get": {
                    "summary": "Get batch job status",
                    "operationId": "getBatchStatus",
                    "tags": ["Batch"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Batch status", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/BatchResponse" } } } },
                        "404": { "description": "Batch not found" }
                    }
                },
                "delete": {
                    "summary": "Cancel a running batch",
                    "operationId": "cancelBatch",
                    "tags": ["Batch"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Batch cancelled" },
                        "404": { "description": "Batch not found" }
                    }
                }
            },

            // ── Video ───────────────────────────────────────────────────────
            "/api/v1/video/upscale": {
                "post": {
                    "summary": "Upscale a video",
                    "operationId": "upscaleVideo",
                    "tags": ["Video"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["video"],
                                    "properties": {
                                        "video": { "type": "string", "format": "binary", "description": "Video file (MP4, AVI, MKV)" },
                                        "scale": { "type": "integer", "enum": [2, 4], "default": 2 },
                                        "model": { "type": "string", "default": "natural" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Video job queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/VideoUpscaleResponse" } } } }
                    }
                }
            },

            // ── Compare ─────────────────────────────────────────────────────
            "/api/v1/compare": {
                "post": {
                    "summary": "Compare upscaling quality across models",
                    "description": "Upscales the same image with multiple models and returns PSNR/SSIM metrics for each.",
                    "operationId": "compareModels",
                    "tags": ["Compare"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/CompareRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Comparison results", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CompareResponse" } } } },
                        "400": { "description": "Invalid request" }
                    }
                }
            },

            // ── Models ──────────────────────────────────────────────────────
            "/api/v1/models": {
                "get": {
                    "summary": "List available upscaling models",
                    "operationId": "listModels",
                    "tags": ["Models"],
                    "security": [],
                    "responses": {
                        "200": { "description": "Model list", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/ModelInfo" } } } } }
                    }
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "List all model versions",
                    "operationId": "listModelVersions",
                    "tags": ["Models"],
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Array of registered model versions",
                            "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/ModelVersion" } } } }
                        }
                    }
                },
                "post": {
                    "summary": "Register a new model version (admin)",
                    "operationId": "createModelVersion",
                    "tags": ["Models"],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateModelVersionRequest" } } }
                    },
                    "responses": {
                        "201": { "description": "Model version created", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelVersion" } } } },
                        "400": { "description": "Invalid request" },
                        "409": { "description": "Model id already exists" }
                    }
                }
            },
            "/v1/models/{id}": {
                "get": {
                    "summary": "Get a model version by id",
                    "operationId": "getModelVersion",
                    "tags": ["Models"],
                    "security": [],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Model version", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelVersion" } } } },
                        "404": { "description": "Model not found" }
                    }
                },
                "delete": {
                    "summary": "Delete a model version (admin)",
                    "operationId": "deleteModelVersion",
                    "tags": ["Models"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "204": { "description": "Model deleted" },
                        "404": { "description": "Model not found" }
                    }
                }
            },

            // ── Jobs ────────────────────────────────────────────────────────
            "/api/v1/job/{id}": {
                "get": {
                    "summary": "Get job status",
                    "operationId": "getJobStatus",
                    "tags": ["Jobs"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Job status", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobStatus" } } } },
                        "404": { "description": "Job not found" }
                    }
                }
            },
            "/api/v1/jobs": {
                "get": {
                    "summary": "List jobs with filtering and pagination",
                    "operationId": "listJobs",
                    "tags": ["Jobs"],
                    "parameters": [
                        { "name": "status", "in": "query", "schema": { "type": "string", "enum": ["pending", "processing", "completed", "failed"] } },
                        { "name": "model", "in": "query", "schema": { "type": "string" } },
                        { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 50 } },
                        { "name": "offset", "in": "query", "schema": { "type": "integer", "default": 0 } }
                    ],
                    "responses": {
                        "200": {
                            "description": "Job list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "jobs": { "type": "array", "items": { "$ref": "#/components/schemas/JobStatus" } },
                                            "total": { "type": "integer" },
                                            "limit": { "type": "integer" },
                                            "offset": { "type": "integer" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/result/{id}": {
                "get": {
                    "summary": "Download upscaled result",
                    "operationId": "getResult",
                    "tags": ["Jobs"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Image binary", "content": { "image/png": { "schema": { "type": "string", "format": "binary" } } } },
                        "404": { "description": "Result not found" }
                    }
                }
            },

            // ── Organizations ───────────────────────────────────────────────
            "/api/v1/orgs": {
                "post": {
                    "summary": "Create an organization",
                    "operationId": "createOrg",
                    "tags": ["Organizations"],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CreateOrgRequest" } } }
                    },
                    "responses": {
                        "201": { "description": "Organization created", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Organization" } } } },
                        "400": { "description": "Invalid request" }
                    }
                },
                "get": {
                    "summary": "List organizations for the authenticated user",
                    "operationId": "listOrgs",
                    "tags": ["Organizations"],
                    "responses": {
                        "200": { "description": "Organization list", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/Organization" } } } } }
                    }
                }
            },
            "/api/v1/orgs/{id}": {
                "get": {
                    "summary": "Get organization details",
                    "operationId": "getOrg",
                    "tags": ["Organizations"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Organization details", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Organization" } } } },
                        "404": { "description": "Organization not found" }
                    }
                }
            },
            "/api/v1/orgs/{id}/members": {
                "post": {
                    "summary": "Add a member to an organization",
                    "operationId": "addOrgMember",
                    "tags": ["Organizations"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AddMemberRequest" } } }
                    },
                    "responses": {
                        "200": { "description": "Member added" },
                        "404": { "description": "Organization not found" }
                    }
                }
            },
            "/api/v1/orgs/{id}/usage": {
                "get": {
                    "summary": "Get organization usage statistics",
                    "operationId": "getOrgUsage",
                    "tags": ["Organizations"],
                    "parameters": [
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } },
                        { "name": "period", "in": "query", "schema": { "type": "string", "enum": ["7d", "30d", "90d"], "default": "30d" } }
                    ],
                    "responses": {
                        "200": { "description": "Usage stats", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OrgUsageResponse" } } } }
                    }
                }
            },

            // ── Analytics ───────────────────────────────────────────────────
            "/api/v1/analytics/usage": {
                "get": {
                    "summary": "Get per-key usage analytics",
                    "operationId": "getUsageAnalytics",
                    "tags": ["Analytics"],
                    "parameters": [
                        { "name": "period", "in": "query", "required": false, "schema": { "type": "string", "enum": ["7d", "30d", "90d"], "default": "30d" } }
                    ],
                    "responses": {
                        "200": { "description": "Usage analytics", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AnalyticsUsageResponse" } } } }
                    }
                }
            },
            "/api/v1/admin/analytics": {
                "get": {
                    "summary": "Get aggregate analytics across all keys (admin only)",
                    "operationId": "getAdminAnalytics",
                    "tags": ["Admin"],
                    "responses": {
                        "200": { "description": "Admin analytics", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/AdminAnalyticsResponse" } } } },
                        "403": { "description": "Not an admin" }
                    }
                }
            },
            "/api/v1/admin/stats": {
                "get": {
                    "summary": "Get system stats (admin only)",
                    "operationId": "getAdminStats",
                    "tags": ["Admin"],
                    "responses": {
                        "200": {
                            "description": "System stats",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "total_keys": { "type": "integer" },
                                            "active_jobs": { "type": "integer" },
                                            "total_completed": { "type": "integer" },
                                            "uptime_seconds": { "type": "integer" }
                                        }
                                    }
                                }
                            }
                        },
                        "403": { "description": "Not an admin" }
                    }
                }
            },
            "/api/v1/admin/batch-stats": {
                "get": {
                    "summary": "Get batch inference engine statistics (admin only)",
                    "operationId": "getBatchInferenceStats",
                    "tags": ["Admin"],
                    "responses": {
                        "200": { "description": "Batch inference stats", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/BatchInferenceStats" } } } },
                        "403": { "description": "Not an admin" }
                    }
                }
            },

            // ── Webhooks ────────────────────────────────────────────────────
            "/api/v1/webhooks": {
                "post": {
                    "summary": "Register a webhook endpoint",
                    "operationId": "registerWebhook",
                    "tags": ["Webhooks"],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/WebhookRegister" } } }
                    },
                    "responses": {
                        "201": { "description": "Webhook registered" }
                    }
                },
                "get": {
                    "summary": "List registered webhooks",
                    "operationId": "listWebhooks",
                    "tags": ["Webhooks"],
                    "responses": {
                        "200": { "description": "Webhook list" }
                    }
                }
            },
            "/api/v1/webhooks/{id}": {
                "delete": {
                    "summary": "Delete a webhook",
                    "operationId": "deleteWebhook",
                    "tags": ["Webhooks"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "204": { "description": "Webhook deleted" },
                        "404": { "description": "Webhook not found" }
                    }
                }
            },
            "/api/v1/webhooks/test": {
                "post": {
                    "summary": "Send a test webhook ping",
                    "operationId": "testWebhook",
                    "tags": ["Webhooks"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["url"],
                                    "properties": {
                                        "url": { "type": "string", "format": "uri" },
                                        "secret": { "type": "string" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Test ping dispatched" }
                    }
                }
            },

            // ── Auth ────────────────────────────────────────────────────────
            "/api/register": {
                "post": {
                    "summary": "Register a new account",
                    "operationId": "register",
                    "tags": ["Auth"],
                    "security": [],
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "type": "object", "required": ["email"], "properties": { "email": { "type": "string", "format": "email" } } } } }
                    },
                    "responses": {
                        "200": { "description": "API key issued" }
                    }
                }
            },

            // ── Detection ───────────────────────────────────────────────────
            "/api/v1/detect": {
                "post": {
                    "summary": "Auto-detect optimal model for an image",
                    "operationId": "detectModel",
                    "tags": ["Image"],
                    "requestBody": {
                        "required": true,
                        "content": { "multipart/form-data": { "schema": { "type": "object", "properties": { "image": { "type": "string", "format": "binary" } } } } }
                    },
                    "responses": {
                        "200": { "description": "Detection result" }
                    }
                }
            },

            // ── System ──────────────────────────────────────────────────────
            "/api/v1/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "healthCheck",
                    "tags": ["System"],
                    "security": [],
                    "responses": {
                        "200": { "description": "Service healthy" }
                    }
                }
            }
        }
    })
    .to_string()
}
