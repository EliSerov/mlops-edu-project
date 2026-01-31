output "project_id" {
  value = module.vpc.project_id
}

output "k8s_cluster_id" {
  value = module.k8s.cluster_id
}

output "kubeconfig_command" {
  value       = "selectel mks kubeconfig ${module.k8s.cluster_id}"
  description = "Команда для получения kubeconfig"
}

output "storage_endpoint" {
  value = module.storage.endpoint
}

output "storage_bucket" {
  value = module.storage.bucket_name
}
