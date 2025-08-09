use srgan_rust::web_server::{ServerConfig, JobStatus, JobInfo};
use std::time::Duration;
use std::path::PathBuf;

#[test]
fn test_server_config_default() {
    let config = ServerConfig::default();
    
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8080);
    assert_eq!(config.max_file_size, 50 * 1024 * 1024); // 50MB
    assert!(config.cache_enabled);
    assert_eq!(config.cache_ttl, Duration::from_secs(3600));
    assert!(config.cors_enabled);
    assert!(config.api_key.is_none());
    assert_eq!(config.rate_limit, Some(60));
    assert!(config.model_path.is_none());
    assert!(config.log_requests);
}

#[test]
fn test_server_config_custom() {
    let mut config = ServerConfig::default();
    config.host = "0.0.0.0".to_string();
    config.port = 3000;
    config.max_file_size = 100 * 1024 * 1024;
    config.cache_enabled = false;
    config.cors_enabled = false;
    config.api_key = Some("secret_key".to_string());
    config.rate_limit = Some(120);
    config.model_path = Some(PathBuf::from("custom_model.rsr"));
    config.log_requests = false;
    
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 3000);
    assert_eq!(config.max_file_size, 100 * 1024 * 1024);
    assert!(!config.cache_enabled);
    assert!(!config.cors_enabled);
    assert_eq!(config.api_key, Some("secret_key".to_string()));
    assert_eq!(config.rate_limit, Some(120));
    assert_eq!(config.model_path, Some(PathBuf::from("custom_model.rsr")));
    assert!(!config.log_requests);
}

#[test]
fn test_job_status_variants() {
    let pending = JobStatus::Pending;
    let processing = JobStatus::Processing;
    let completed = JobStatus::Completed;
    let failed = JobStatus::Failed("Error message".to_string());
    
    match pending {
        JobStatus::Pending => assert!(true),
        _ => assert!(false),
    }
    
    match processing {
        JobStatus::Processing => assert!(true),
        _ => assert!(false),
    }
    
    match completed {
        JobStatus::Completed => assert!(true),
        _ => assert!(false),
    }
    
    match failed {
        JobStatus::Failed(msg) => assert_eq!(msg, "Error message"),
        _ => assert!(false),
    }
}

#[test]
fn test_job_info_creation() {
    let job = JobInfo {
        id: "job_123".to_string(),
        status: JobStatus::Pending,
        created_at: 1000,
        updated_at: 1000,
        result_url: None,
        error: None,
    };
    
    assert_eq!(job.id, "job_123");
    assert!(matches!(job.status, JobStatus::Pending));
    assert_eq!(job.created_at, 1000);
    assert_eq!(job.updated_at, 1000);
    assert!(job.result_url.is_none());
    assert!(job.error.is_none());
}

#[test]
fn test_job_info_with_result() {
    let job = JobInfo {
        id: "job_456".to_string(),
        status: JobStatus::Completed,
        created_at: 1000,
        updated_at: 2000,
        result_url: Some("/api/result/job_456".to_string()),
        error: None,
    };
    
    assert_eq!(job.id, "job_456");
    assert!(matches!(job.status, JobStatus::Completed));
    assert_eq!(job.created_at, 1000);
    assert_eq!(job.updated_at, 2000);
    assert_eq!(job.result_url, Some("/api/result/job_456".to_string()));
    assert!(job.error.is_none());
}

#[test]
fn test_job_info_with_error() {
    let job = JobInfo {
        id: "job_789".to_string(),
        status: JobStatus::Failed("Processing error".to_string()),
        created_at: 1000,
        updated_at: 1500,
        result_url: None,
        error: Some("Processing error".to_string()),
    };
    
    assert_eq!(job.id, "job_789");
    assert!(matches!(job.status, JobStatus::Failed(_)));
    assert_eq!(job.created_at, 1000);
    assert_eq!(job.updated_at, 1500);
    assert!(job.result_url.is_none());
    assert_eq!(job.error, Some("Processing error".to_string()));
}