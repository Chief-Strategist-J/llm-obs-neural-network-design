resource "aws_s3_bucket" "checkpoints" {
  bucket = "${var.environment}-${var.service_name}-checkpoints"

  tags = {
    Environment = var.environment
    Service     = var.service_name
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_versioning" "checkpoint_versioning" {
  bucket = aws_s3_bucket.checkpoints.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "checkpoint_crypto" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "checkpoint_bucket_id" {
  value = aws_s3_bucket.checkpoints.id
}
