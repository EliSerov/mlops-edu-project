output "project_id" {
  value = selectel_vpc_project_v2.main.id
}

output "network_id" {
  value = openstack_networking_network_v2.main.id
}

output "subnet_id" {
  value = openstack_networking_subnet_v2.main.id
}

output "security_group_id" {
  value = openstack_networking_secgroup_v2.k8s.id
}
