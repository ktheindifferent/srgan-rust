// Package srgan provides an idiomatic Go client for the SRGAN-Rust REST API.
//
// The Client wraps all HTTP communication and provides both synchronous and
// asynchronous upscaling workflows.  Every method that performs I/O accepts a
// [context.Context] for cancellation and deadline propagation.
//
// Quick start:
//
//	client := srgan.NewClient("http://localhost:8080", "sk-...")
//
//	// Synchronous single-image upscale
//	data, err := client.Upscale(ctx, "photo.jpg", 4)
//
//	// Async workflow
//	job, _ := client.UpscaleAsync(ctx, "photo.jpg", nil)
//	job, _ = client.WaitForJob(ctx, job.ID)
//	_ = client.DownloadResult(ctx, job.ID, "photo_4x.png")
package srgan

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

// APIError represents an error response from the SRGAN API.
type APIError struct {
	// StatusCode is the HTTP status code returned by the server.
	StatusCode int
	// Message is the human-readable error description.
	Message string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("srgan: HTTP %d: %s", e.StatusCode, e.Message)
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

// UpscaleOptions configures an upscale request.
// All fields are optional; zero values are omitted from the JSON payload.
type UpscaleOptions struct {
	// Model selects the upscaling model: "natural", "anime", "bilinear", etc.
	Model string `json:"model,omitempty"`
	// OutputFormat is the desired output encoding: "png", "jpeg", or "webp".
	OutputFormat string `json:"output_format,omitempty"`
	// Quality sets the JPEG/WebP quality (1-100, ignored for PNG).
	Quality int `json:"quality,omitempty"`
}

// Job describes the state of an upscaling job.
type Job struct {
	// ID is the unique job identifier.
	ID string `json:"id"`
	// Status is one of: "Pending", "Processing", "Completed", "Failed".
	Status string `json:"status"`
	// ResultData holds the base64-encoded result image when Status is "Completed".
	ResultData string `json:"result_data,omitempty"`
	// Error contains the failure reason when Status is "Failed".
	Error string `json:"error,omitempty"`
	// Model used for this job.
	Model string `json:"model,omitempty"`
	// InputSize is the original dimensions "WxH".
	InputSize string `json:"input_size,omitempty"`
	// OutputSize is the upscaled dimensions "WxH".
	OutputSize string `json:"output_size,omitempty"`
	// ScaleFactor applied.
	ScaleFactor int `json:"scale_factor,omitempty"`
	// Progress percentage (0-100) for long-running jobs.
	Progress int `json:"progress,omitempty"`
}

// IsDone reports whether the job has reached a terminal state.
func (j *Job) IsDone() bool {
	switch j.Status {
	case "Completed", "completed", "Failed", "failed", "cancelled":
		return true
	}
	return false
}

// HealthResponse contains the server health information.
type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version,omitempty"`
	Uptime  int64  `json:"uptime,omitempty"`
}

// ModelInfo describes an available upscaling model.
type ModelInfo struct {
	Name         string `json:"name"`
	Description  string `json:"description,omitempty"`
	ScaleFactors []int  `json:"scale_factors,omitempty"`
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

// Client is an idiomatic Go client for the SRGAN-Rust REST API.
//
// Create one with [NewClient] and reuse it for multiple requests.
type Client struct {
	// BaseURL is the root URL of the SRGAN server (e.g. "http://localhost:8080").
	BaseURL string
	// APIKey is sent as the X-API-Key header on every request.
	APIKey string
	// HTTPClient is the underlying HTTP client. If nil, a default client with
	// a 120-second timeout is used.
	HTTPClient *http.Client
}

// NewClient creates a [Client] configured with the given base URL and API key.
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

// ---------------------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------------------

// Upscale reads the image at inputPath, sends it to the synchronous upscale
// endpoint with the given scale factor, and returns the resulting image bytes.
//
// Pass 0 for scale to use the server default (4x).
func (c *Client) Upscale(ctx context.Context, inputPath string, scale int) (*Job, error) {
	encoded, err := encodeImageFile(inputPath)
	if err != nil {
		return nil, err
	}

	body := map[string]interface{}{
		"image_data": encoded,
	}
	if scale > 0 {
		body["scale_factor"] = scale
	}

	var resp struct {
		Success    bool   `json:"success"`
		ImageData  string `json:"image_data"`
		Error      string `json:"error"`
		Metadata   struct {
			OriginalSize   [2]uint32 `json:"original_size"`
			UpscaledSize   [2]uint32 `json:"upscaled_size"`
			ProcessingTime uint64    `json:"processing_time_ms"`
			Format         string    `json:"format"`
			ModelUsed      string    `json:"model_used"`
		} `json:"metadata"`
	}
	if err := c.postJSON(ctx, "/api/v1/upscale", body, &resp); err != nil {
		return nil, err
	}
	if !resp.Success {
		return nil, &APIError{StatusCode: 400, Message: resp.Error}
	}

	return &Job{
		Status:     "Completed",
		ResultData: resp.ImageData,
		Model:      resp.Metadata.ModelUsed,
		InputSize:  fmt.Sprintf("%dx%d", resp.Metadata.OriginalSize[0], resp.Metadata.OriginalSize[1]),
		OutputSize: fmt.Sprintf("%dx%d", resp.Metadata.UpscaledSize[0], resp.Metadata.UpscaledSize[1]),
		ScaleFactor: scale,
	}, nil
}

// UpscaleAsync submits an image for asynchronous upscaling and returns the
// initial [Job] with its ID. Use [Client.GetJob] or [Client.WaitForJob] to
// track progress.
func (c *Client) UpscaleAsync(ctx context.Context, inputPath string, opts *UpscaleOptions) (*Job, error) {
	encoded, err := encodeImageFile(inputPath)
	if err != nil {
		return nil, err
	}

	body := map[string]interface{}{
		"image_data": encoded,
	}
	if opts != nil {
		if opts.Model != "" {
			body["model"] = opts.Model
		}
		if opts.OutputFormat != "" {
			body["output_format"] = opts.OutputFormat
		}
		if opts.Quality > 0 {
			body["quality"] = opts.Quality
		}
	}

	var result struct {
		JobID string `json:"job_id"`
	}
	if err := c.postJSON(ctx, "/api/v1/upscale/async", body, &result); err != nil {
		return nil, err
	}
	return &Job{ID: result.JobID, Status: "Pending"}, nil
}

// GetJob retrieves the current status of a job by its ID.
func (c *Client) GetJob(ctx context.Context, jobID string) (*Job, error) {
	var raw struct {
		ID         string  `json:"id"`
		Status     string  `json:"status"`
		ResultData *string `json:"result_data"`
		Error      *string `json:"error"`
		Model      *string `json:"model"`
		InputSize  *string `json:"input_size"`
		OutputSize *string `json:"output_size"`
		ScaleFactor *int   `json:"scale_factor"`
		Progress   *int    `json:"progress"`
	}
	if err := c.getJSON(ctx, "/api/v1/job/"+jobID, &raw); err != nil {
		return nil, err
	}
	job := &Job{
		ID:     raw.ID,
		Status: raw.Status,
	}
	if raw.ResultData != nil {
		job.ResultData = *raw.ResultData
	}
	if raw.Error != nil {
		job.Error = *raw.Error
	}
	if raw.Model != nil {
		job.Model = *raw.Model
	}
	if raw.InputSize != nil {
		job.InputSize = *raw.InputSize
	}
	if raw.OutputSize != nil {
		job.OutputSize = *raw.OutputSize
	}
	if raw.ScaleFactor != nil {
		job.ScaleFactor = *raw.ScaleFactor
	}
	if raw.Progress != nil {
		job.Progress = *raw.Progress
	}
	return job, nil
}

// WaitForJob polls the job status every 2 seconds until it reaches a terminal
// state (Completed or Failed). The context can be used to set a deadline or
// cancel the polling loop.
//
// Returns the final [Job] on success. If the job failed, the returned error
// wraps the failure reason.
func (c *Client) WaitForJob(ctx context.Context, jobID string) (*Job, error) {
	const pollInterval = 2 * time.Second

	for {
		job, err := c.GetJob(ctx, jobID)
		if err != nil {
			return nil, err
		}
		if job.IsDone() {
			if job.Status == "Failed" || job.Status == "failed" {
				return job, fmt.Errorf("srgan: job %s failed: %s", jobID, job.Error)
			}
			return job, nil
		}

		select {
		case <-ctx.Done():
			return job, ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}

// DownloadResult fetches the result of a completed job and writes it to
// outputPath. The file is created with mode 0644.
func (c *Client) DownloadResult(ctx context.Context, jobID, outputPath string) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.BaseURL+"/api/v1/result/"+jobID, nil)
	if err != nil {
		return err
	}
	c.setAuth(req)

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("srgan: download request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return parseAPIError(resp)
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}

// Health returns the server health status.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var h HealthResponse
	if err := c.getJSON(ctx, "/api/v1/health", &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// ListModels returns the list of models available on the server.
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	var wrapper struct {
		Models []ModelInfo `json:"models"`
	}
	if err := c.getJSON(ctx, "/api/v1/models", &wrapper); err != nil {
		return nil, err
	}
	return wrapper.Models, nil
}

// CancelJob cancels a queued or in-progress job.
func (c *Client) CancelJob(ctx context.Context, jobID string) error {
	req, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/api/v1/jobs/"+jobID+"/cancel", nil)
	if err != nil {
		return err
	}
	c.setAuth(req)

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("srgan: cancel request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return parseAPIError(resp)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (c *Client) setAuth(req *http.Request) {
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
}

func (c *Client) postJSON(ctx context.Context, path string, payload interface{}, dest interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("srgan: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	c.setAuth(req)

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("srgan: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return parseAPIError(resp)
	}

	return json.NewDecoder(resp.Body).Decode(dest)
}

func (c *Client) getJSON(ctx context.Context, path string, dest interface{}) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.BaseURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	c.setAuth(req)

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("srgan: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return parseAPIError(resp)
	}

	return json.NewDecoder(resp.Body).Decode(dest)
}

func parseAPIError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	msg := string(body)
	var parsed struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &parsed) == nil && parsed.Error != "" {
		msg = parsed.Error
	}
	return &APIError{StatusCode: resp.StatusCode, Message: msg}
}

func encodeImageFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("srgan: failed to read image %q: %w", path, err)
	}
	return base64.StdEncoding.EncodeToString(data), nil
}
