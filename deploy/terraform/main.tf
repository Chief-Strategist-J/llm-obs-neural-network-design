module "security" {
  source       = "./modules/security"
  environment  = var.environment
  service_name = var.service_name
  vpc_id       = var.vpc_id
}

module "stateful" {
  source            = "./modules/stateful"
  environment       = var.environment
  service_name      = var.service_name
  db_instance_class = var.db_instance_class
}

module "stateless" {
  source       = "./modules/stateless"
  environment  = var.environment
  service_name = var.service_name
  min_replicas = var.min_replicas
  max_replicas = var.max_replicas
}
