terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.51.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- APIs ---
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "firestore.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudtasks.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com"
  ])
  service            = each.key
  disable_on_destroy = false
}

# --- Artifact Registry ---
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "caustic-repo"
  description   = "Docker repository for Caustic Design"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# --- Service Accounts ---
resource "google_service_account" "runner" {
  account_id   = "caustic-runner-sa"
  display_name = "Caustic Design Job Runner"
}

resource "google_service_account" "api" {
  account_id   = "caustic-api-sa"
  display_name = "Caustic Design API"
}

# --- IAM Roles for Runner ---
resource "google_project_iam_member" "runner_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_project_iam_member" "runner_datastore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_project_iam_member" "runner_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

# --- IAM Roles for API ---
resource "google_project_iam_member" "api_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_tasks_enqueuer" {
  project = var.project_id
  role    = "roles/cloudtasks.enqueuer"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_service_account_iam_member" "api_act_as_runner" {
  service_account_id = google_service_account.runner.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}


# --- Storage Buckets ---
resource "google_storage_bucket" "inputs" {
  name          = "${var.project_id}-caustic-inputs"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "outputs" {
  name          = "${var.project_id}-caustic-outputs"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

# --- Firestore (App Engine Required) ---
# Note: Enabling Firestore often creates an App Engine app implicitly or explicitely.
# Terraform support for native mode database creation is via `google_firestore_database`
# but usually assumes the project is already in Firestore Datastore mode.
# We will use the `google_project_service` to enable it, 
# and assume the user creates the default database or we can define it.
# Recent TF versions support `google_firestore_database`.

resource "google_firestore_database" "database" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  depends_on  = [google_project_service.apis]
}

# --- Cloud Run Job ---
resource "google_cloud_run_v2_job" "default" {
  name     = "caustic-job"
  location = var.region
  
  template {
    template {
      containers {
        # Using a placeholder image initially or specific one if known.
        # Ideally this is replaced by CI/CD.
        image = "us-docker.pkg.dev/cloudrun/container/hello" 
        resources {
          limits = {
            cpu    = "4"
            memory = "8Gi"
          }
        }
        env {
           name = "CLOUD_LOG_NAME"
           value = "caustic-job-log"
        }
      }
      service_account = google_service_account.runner.email
      timeout = "3600s" # 1 hour
    }
  }
  
  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image, # Ignore image changes from external deployments
    ]
  }

  depends_on = [google_project_service.apis]
}

# --- Cloud Tasks Queue ---
resource "google_cloud_tasks_queue" "default" {
  name     = "caustic-queue"
  location = var.region
  depends_on = [google_project_service.apis]
}
