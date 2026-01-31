variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "network_id" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "k8s_version" {
  type = string
}

variable "environment" {
  type = string
}

variable "node_groups" {
  type = map(object({
    count       = number
    flavor      = string
    volume_size = number
    labels      = map(string)
  }))
}
