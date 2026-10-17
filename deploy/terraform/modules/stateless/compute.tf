resource "kubernetes_deployment" "service_app" {
  metadata {
    name = var.service_name
    labels = {
      app       = var.service_name
      managed_by = "terraform"
    }
  }

  spec {
    replicas = var.min_replicas

    selector {
      match_labels = {
        app = var.service_name
      }
    }

    template {
      metadata {
        labels = {
          app = var.service_name
        }
      }

      spec {
        container {
          image = "ghcr.io/scaibu/${var.service_name}:latest"
          name  = var.service_name

          port {
            container_port = 8080
          }

          resources {
            limits = {
              cpu    = "1000m"
              memory = "2Gi"
            }
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 15
            period_seconds        = 10
          }
        }
      }
    }
  }
}
