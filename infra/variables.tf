variable "selectel_token" {
  type        = string
  description = "API токен Selectel"
  sensitive   = true
}

variable "selectel_account_id" {
  type        = string
  description = "ID аккаунта Selectel (домен)"
}

variable "openstack_user" {
  type      = string
  sensitive = true
}

variable "openstack_password" {
  type      = string
  sensitive = true
}

variable "project_name" {
  type    = string
  default = "credit-scoring"
}

variable "region" {
  type    = string
  default = "ru-3"
}

variable "environment" {
  type    = string
  default = "staging"
}

variable "k8s_version" {
  type    = string
  default = "1.28"
}

variable "node_groups" {
  type = map(object({
    count       = number
    flavor      = string
    volume_size = number
    labels      = map(string)
  }))
  default = {
    cpu = {
      count       = 2
      flavor      = "SL1.2-4096"  # 2 vCPU, 4GB RAM
      volume_size = 30
      labels      = { workload = "general" }
    }
    gpu = {
      count       = 0  # отключено по умолчанию, дорого
      flavor      = "GPU1.A4000-1-24-64"
      volume_size = 50
      labels      = { workload = "gpu" }
    }
  }
}
