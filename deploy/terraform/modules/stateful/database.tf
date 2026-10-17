resource "aws_db_instance" "database" {
  identifier            = "${var.environment}-${var.service_name}-db"
  allocated_storage     = 50
  max_allocated_storage = 500
  engine                = "postgres"
  engine_version        = "15.4"
  instance_class        = var.db_instance_class
  db_name               = "policy_orchestration"
  username              = "alloy_admin"
  password              = "alloysecret"
  skip_final_snapshot   = var.environment == "prod" ? false : true
  deletion_protection   = var.environment == "prod" ? true : false

  lifecycle {
    prevent_destroy = false
  }

  tags = {
    Environment = var.environment
    Service     = var.service_name
    ManagedBy   = "Terraform"
  }
}

output "db_endpoint" {
  value = aws_db_instance.database.endpoint
}
