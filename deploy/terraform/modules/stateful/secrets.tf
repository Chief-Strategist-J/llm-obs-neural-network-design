resource "aws_secretsmanager_secret" "db_credentials" {
  name                    = "${var.environment}/${var.service_name}/database"
  recovery_window_in_days = 0

  tags = {
    Environment = var.environment
    Service     = var.service_name
    ManagedBy   = "Terraform"
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials_val" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = "alloy_admin"
    password = "alloysecret"
    database = "policy_orchestration"
  })
}

output "secret_arn" {
  value = aws_secretsmanager_secret.db_credentials.arn
}
