resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.environment}-${var.service_name}-cache"
  engine               = "redis"
  node_type            = "cache.t4g.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379

  tags = {
    Environment = var.environment
    Service     = var.service_name
    ManagedBy   = "Terraform"
  }
}

output "cache_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}
