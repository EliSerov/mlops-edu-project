# Selectel MKS (Managed Kubernetes) + инфраструктура
# Модули: vpc, k8s, storage, monitoring

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    selectel = {
      source  = "selectel/selectel"
      version = "~> 5.0"
    }
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.54"
    }
  }
  
  backend "s3" {
    # конфигурируется через backend.hcl
    key = "credit-scoring/terraform.tfstate"
  }
}

provider "selectel" {
  token = var.selectel_token
}

provider "openstack" {
  auth_url    = "https://cloud.api.selcloud.ru/identity/v3"
  domain_name = var.selectel_account_id
  tenant_id   = module.vpc.project_id
  user_name   = var.openstack_user
  password    = var.openstack_password
  region      = var.region
}

module "vpc" {
  source = "./modules/vpc"
  
  project_name = var.project_name
  region       = var.region
}

module "k8s" {
  source = "./modules/k8s"
  
  project_id        = module.vpc.project_id
  region            = var.region
  network_id        = module.vpc.network_id
  subnet_id         = module.vpc.subnet_id
  k8s_version       = var.k8s_version
  environment       = var.environment
  
  node_groups = var.node_groups
}

module "storage" {
  source = "./modules/storage"
  
  project_id   = module.vpc.project_id
  bucket_name  = "${var.project_name}-data"
}

module "monitoring" {
  source = "./modules/monitoring"
  
  project_id = module.vpc.project_id
  region     = var.region
}
