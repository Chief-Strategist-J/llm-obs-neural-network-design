resource "aws_kms_key" "service_key" {
  description             = "KMS key for ${var.environment}-${var.service_name} encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Environment = var.environment
    Service     = var.service_name
    ManagedBy   = "Terraform"
  }
}

resource "aws_kms_alias" "service_key_alias" {
  name          = "alias/${var.environment}-${var.service_name}-key"
  target_key_id = aws_kms_key.service_key.key_id
}

output "kms_key_arn" {
  value = aws_kms_key.service_key.arn
}
