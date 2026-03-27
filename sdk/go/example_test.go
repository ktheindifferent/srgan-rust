package srgan_test

import (
	"context"
	"fmt"
	"log"
	"time"

	srgan "github.com/user/srgan-rust/sdk/go"
)

// ExampleClient_Upscale demonstrates synchronous image upscaling.
func ExampleClient_Upscale() {
	client := srgan.NewClient("http://localhost:8080", "sk-test-key")
	ctx := context.Background()

	job, err := client.Upscale(ctx, "photo.jpg", 4)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Upscaled %s -> %s using %s\n", job.InputSize, job.OutputSize, job.Model)
}

// ExampleClient_WaitForJob demonstrates the async workflow: submit, poll, download.
func ExampleClient_WaitForJob() {
	client := srgan.NewClient("http://localhost:8080", "sk-test-key")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Submit async job
	job, err := client.UpscaleAsync(ctx, "photo.jpg", &srgan.UpscaleOptions{
		Model: "anime",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Job submitted: %s\n", job.ID)

	// Poll until done
	job, err = client.WaitForJob(ctx, job.ID)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Job %s completed\n", job.ID)

	// Download result
	if err := client.DownloadResult(ctx, job.ID, "photo_4x.png"); err != nil {
		log.Fatal(err)
	}
}

// ExampleClient_GetJob demonstrates checking a single job's status.
func ExampleClient_GetJob() {
	client := srgan.NewClient("http://localhost:8080", "sk-test-key")
	ctx := context.Background()

	job, err := client.GetJob(ctx, "job-abc123")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Job %s: status=%s progress=%d%%\n", job.ID, job.Status, job.Progress)
}
