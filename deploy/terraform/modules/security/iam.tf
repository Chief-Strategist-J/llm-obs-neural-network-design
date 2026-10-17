resource "aws_iam_role" "service_role" {
  name = "${var.environment}-${var.service_name}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
    Service     = var.service_name
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_policy" "checkpoint_access" {
  name        = "${var.environment}-${var.service_name}-s3-policy"
  description = "Allows read/write access to model checkpoint bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.environment}-${var.service_name}-checkpoints",
          "arn:aws:s3:::${var.environment}-${var.service_name}-checkpoints/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "checkpoint_attach" {
  role       = aws_iam_role.service_role.name
  policy_arn = aws_iam_policy.checkpoint_access.arn
}

output "service_role_arn" {
  value = aws_iam_role.service_role.arn
}
