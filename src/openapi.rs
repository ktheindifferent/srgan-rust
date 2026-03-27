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

/// Returns the full OpenAPI 3.0 JSON specification including model versioning
/// CRUD endpoints and all existing endpoints.
pub fn render_openapi_json() -> String {
    serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "SRGAN Image Super-Resolution API",
            "description": "AI-powered image and video upscaling service built in Rust.",
            "version": "1.1.0",
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
            { "name": "Video", "description": "Video upscaling endpoints" },
            { "name": "Batch", "description": "Batch processing endpoints" },
            { "name": "Jobs", "description": "Job status and result retrieval" },
            { "name": "Models", "description": "Model listing and versioning" },
            { "name": "Webhooks", "description": "Webhook management" },
            { "name": "Auth", "description": "Authentication and registration" },
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
                        "status": { "type": "string" },
                        "progress": { "type": "integer", "minimum": 0, "maximum": 100 },
                        "created_at": { "type": "string", "format": "date-time" }
                    }
                },
                "BatchResponse": {
                    "type": "object",
                    "properties": {
                        "batch_id": { "type": "string" },
                        "total": { "type": "integer" },
                        "status": { "type": "string" }
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
                        "max_scale": { "type": "integer" }
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
                                        "model": { "type": "string", "enum": ["natural", "anime", "waifu2x", "real-esrgan"], "default": "natural" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Upscale complete", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UpscaleResponse" } } } },
                        "400": { "description": "Invalid request" },
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
            "/api/v1/batch": {
                "post": {
                    "summary": "Batch upscale multiple images",
                    "operationId": "batchUpscale",
                    "tags": ["Batch"],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["images"],
                                    "properties": {
                                        "images": { "type": "array", "items": { "type": "string", "format": "binary" } },
                                        "scale": { "type": "integer", "default": 4 },
                                        "model": { "type": "string", "default": "natural" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": { "description": "Batch queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/BatchResponse" } } } }
                    }
                }
            },
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
            },
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
            }
        }
    })
    .to_string()
}
