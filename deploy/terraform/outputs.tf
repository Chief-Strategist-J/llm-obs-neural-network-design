output "service_name" {
  description = "The registered microservice name"
  value       = var.service_name
}

output "environment" {
  description = "Current deployment environment"
  value       = var.environment
}

output "checkpoint_bucket_name" {
  description = "S3 bucket name for neural network checkpoints"
  value       = module.stateful.checkpoint_bucket_id
}

output "security_group_id" {
  description = "Security group attached to the compute workloads"
  value       = module.security.service_security_group_id
}
