resource "kubernetes_service" "service_ingress" {
  metadata {
    name = var.service_name
  }

  spec {
    selector = {
      app = var.service_name
    }

    port {
      port        = 8080
      target_port = 8080
      name        = "http"
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_ingress_v1" "ingress" {
  metadata {
    name = "${var.service_name}-ingress"
    annotations = {
      "kubernetes.io/ingress.class"                 = "nginx"
      "nginx.ingress.kubernetes.io/proxy-body-size" = "64m"
    }
  }

  spec {
    rule {
      host = "${var.service_name}.llmobs.internal"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service.service_ingress.metadata[0].name
              port {
                number = 8080
              }
            }
          }
        }
      }
    }
  }
}
