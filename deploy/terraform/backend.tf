terraform {
  backend "s3" {
    bucket         = "llmobs-terraform-state"
    key            = "services/neural-network-design/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "llmobs-terraform-locks"
  }
}
