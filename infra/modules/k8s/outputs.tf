output "cluster_id" {
  value = selectel_mks_cluster_v1.main.id
}

output "cluster_name" {
  value = selectel_mks_cluster_v1.main.name
}

output "kube_api_ip" {
  value = selectel_mks_cluster_v1.main.kube_api_ip
}
