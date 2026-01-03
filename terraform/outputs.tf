output "artifact_registry_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.name}"
}

output "input_bucket" {
  value = google_storage_bucket.inputs.name
}

output "output_bucket" {
  value = google_storage_bucket.outputs.name
}

output "job_name" {
  value = google_cloud_run_v2_job.default.name
}

output "queue_name" {
  value = google_cloud_tasks_queue.default.name
}

output "runner_service_account" {
  value = google_service_account.runner.email
}
