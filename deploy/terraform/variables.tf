variable "environment" {
  description = "Target deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "service_name" {
  description = "Name of the microservice"
  type        = string
  default     = "neural-network-design"
}

variable "vpc_id" {
  description = "VPC ID where service and datastores reside"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for compute and data workloads"
  type        = list(string)
  default     = []
}

variable "min_replicas" {
  description = "Minimum compute replica count"
  type        = number
  default     = 2
}

variable "max_replicas" {
  description = "Maximum compute replica count"
  type        = number
  default     = 10
}

variable "db_instance_class" {
  description = "RDS/AlloyDB database instance class"
  type        = string
  default     = "db.r6g.large"
}
