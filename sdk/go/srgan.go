// Package srgan provides a Go client for the SRGAN-Rust REST API.
//
// Usage:
//
//	client := srgan.NewClient("http://localhost:8080", "sk-...")
//
//	// Synchronous upscale
//	data, err := client.Upscale("photo.jpg", &srgan.UpscaleOptions{Scale: 4, Model: "natural"})
//
//	// Async workflow
//	jobID, _ := client.UpscaleAsync("photo.jpg", nil)
//	status, _ := client.WaitForJob(jobID)
//	_ = client.DownloadResult(jobID, "photo_4x.png")
package srgan

import (
	"bytes"
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
type UpscaleOptions struct {
	// Scale is the upscaling factor (2, 4, or 8). Defaults to 4 if zero.
	Scale int `json:"scale_factor,omitempty"`
	// Model selects the upscaling model: "natural", "anime", or "bilinear".
	Model string `json:"model,omitempty"`
	// OutputFormat is the desired output encoding: "png", "jpeg", or "webp".
	OutputFormat string `json:"output_format,omitempty"`
	// Quality sets the JPEG/WebP quality (1-100, ignored for PNG).
	Quality int `json:"quality,omitempty"`
}

// JobStatus describes the current state of an async upscaling job.
type JobStatus struct {
	// JobID is the unique identifier for the job.
	JobID string `json:"job_id"`
	// Status is one of: "queued", "processing", "completed", "failed", "cancelled".
	Status string `json:"status"`
	// Progress is the completion percentage (0-100).
	Progress int `json:"progress_pct"`
	// ResultData holds the base64-encoded result image when status is "completed".
	ResultData string `json:"result_data,omitempty"`
	// Error contains the failure reason when status is "failed".
	Error string `json:"error,omitempty"`
}

// IsDone reports whether the job has reached a terminal state.
func (j *JobStatus) IsDone() bool {
	switch j.Status {
	case "completed", "failed", "cancelled":
		return true
	}
	return false
}

// HealthResponse contains the server health information.
type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version,omitempty"`
	Uptime  string `json:"uptime,omitempty"`
}

// ModelInfo describes an available upscaling model.
type ModelInfo struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	ScaleFactors []int `json:"scale_factors,omitempty"`
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

// Client is a thin wrapper around the SRGAN-Rust REST API.
type Client struct {
	// BaseURL is the root URL of the SRGAN server (e.g. "http://localhost:8080").
	BaseURL string
	// APIKey is sent as the X-API-Key header on every request.
	APIKey string
	// HTTPClient is the underlying HTTP client. If nil, http.DefaultClient is used.
	HTTPClient *http.Client
}

// NewClient creates a Client configured with the given base URL and API key.
// It uses http.DefaultClient for requests; set client.HTTPClient to override.
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: 120 * time.Second},
	}
}

// httpClient returns the configured HTTP client or the default.
func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

// ---------------------------------------------------------------------------
// Public methods
// ---------------------------------------------------------------------------

// Upscale reads the image at imagePath, sends it to the synchronous upscale
// endpoint, and returns the resulting image bytes.
// Pass nil for opts to use server defaults.
func (c *Client) Upscale(imagePath string, opts *UpscaleOptions) ([]byte, error) {
	encoded, err := encodeImageFile(imagePath)
	if err != nil {
		return nil, err
	}

	body := buildUpscalePayload(encoded, opts)

	var result struct {
		ImageBase64 string `json:"image_base64"`
	}
	if err := c.postJSON("/api/v1/upscale", body, &result); err != nil {
		return nil, err
	}

	return base64.StdEncoding.DecodeString(result.ImageBase64)
}

// UpscaleAsync submits an image for asynchronous upscaling and returns the
// job ID. Use GetJob or WaitForJob to track progress.
func (c *Client) UpscaleAsync(imagePath string, opts *UpscaleOptions) (string, error) {
	encoded, err := encodeImageFile(imagePath)
	if err != nil {
		return "", err
	}

	body := buildUpscalePayload(encoded, opts)

	var result struct {
		JobID string `json:"job_id"`
	}
	if err := c.postJSON("/api/v1/upscale/async", body, &result); err != nil {
		return "", err
	}

	return result.JobID, nil
}

// GetJob retrieves the current status of a job.
func (c *Client) GetJob(jobID string) (*JobStatus, error) {
	var status JobStatus
	if err := c.getJSON("/api/v1/job/"+jobID, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

// WaitForJob polls the job status every 2 seconds until it reaches a terminal
// state (completed, failed, or cancelled). It returns an error if the job
// failed or was cancelled.
func (c *Client) WaitForJob(jobID string) (*JobStatus, error) {
	const pollInterval = 2 * time.Second

	for {
		status, err := c.GetJob(jobID)
		if err != nil {
			return nil, err
		}

		if status.IsDone() {
			if status.Status == "failed" {
				return status, fmt.Errorf("srgan: job %s failed: %s", jobID, status.Error)
			}
			if status.Status == "cancelled" {
				return status, fmt.Errorf("srgan: job %s was cancelled", jobID)
			}
			return status, nil
		}

		time.Sleep(pollInterval)
	}
}

// DownloadResult fetches the result of a completed job and writes it to
// outputPath. The file is created with mode 0644.
func (c *Client) DownloadResult(jobID, outputPath string) error {
	req, err := http.NewRequest("GET", c.BaseURL+"/api/v1/result/"+jobID, nil)
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
func (c *Client) Health() (*HealthResponse, error) {
	var h HealthResponse
	if err := c.getJSON("/api/v1/health", &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// ListModels returns the list of models available on the server.
func (c *Client) ListModels() ([]ModelInfo, error) {
	var wrapper struct {
		Models []ModelInfo `json:"models"`
	}
	if err := c.getJSON("/api/v1/models", &wrapper); err != nil {
		return nil, err
	}
	return wrapper.Models, nil
}

// CancelJob cancels a queued or in-progress job.
func (c *Client) CancelJob(jobID string) error {
	req, err := http.NewRequest("POST", c.BaseURL+"/api/jobs/"+jobID+"/cancel", nil)
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

// setAuth applies the X-API-Key header to a request.
func (c *Client) setAuth(req *http.Request) {
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
}

// postJSON sends a POST request with a JSON body and decodes the response
// into dest.
func (c *Client) postJSON(path string, payload interface{}, dest interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("srgan: failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", c.BaseURL+path, bytes.NewReader(data))
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

// getJSON sends a GET request and decodes the JSON response into dest.
func (c *Client) getJSON(path string, dest interface{}) error {
	req, err := http.NewRequest("GET", c.BaseURL+path, nil)
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

// parseAPIError reads an error response body and returns an *APIError.
func parseAPIError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	msg := string(body)
	var parsed struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &parsed) == nil && parsed.Error != "" {
		msg = parsed.Error
	}

	return &APIError{
		StatusCode: resp.StatusCode,
		Message:    msg,
	}
}

// encodeImageFile reads a file and returns its contents as a base64 string.
func encodeImageFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("srgan: failed to read image %q: %w", path, err)
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

// buildUpscalePayload constructs the JSON body for upscale endpoints.
func buildUpscalePayload(imageBase64 string, opts *UpscaleOptions) map[string]interface{} {
	body := map[string]interface{}{
		"image_data": imageBase64,
	}
	if opts != nil {
		if opts.Scale > 0 {
			body["scale_factor"] = opts.Scale
		}
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
	return body
}
