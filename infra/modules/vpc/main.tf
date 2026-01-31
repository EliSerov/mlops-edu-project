# Модуль VPC — проект + настройка сети

resource "selectel_vpc_project_v2" "main" {
  name = var.project_name
}

resource "openstack_networking_network_v2" "main" {
  name           = "${var.project_name}-network"
  admin_state_up = true
}

resource "openstack_networking_subnet_v2" "main" {
  name       = "${var.project_name}-subnet"
  network_id = openstack_networking_network_v2.main.id
  cidr       = "10.0.0.0/24"
  ip_version = 4
  
  dns_nameservers = ["188.93.16.19", "188.93.17.19"]  # selectel dns
}

resource "openstack_networking_router_v2" "main" {
  name                = "${var.project_name}-router"
  external_network_id = data.openstack_networking_network_v2.external.id
}

resource "openstack_networking_router_interface_v2" "main" {
  router_id = openstack_networking_router_v2.main.id
  subnet_id = openstack_networking_subnet_v2.main.id
}

data "openstack_networking_network_v2" "external" {
  external = true
}

# security group для k8s нод
resource "openstack_networking_secgroup_v2" "k8s" {
  name        = "${var.project_name}-k8s-sg"
  description = "Security group для K8s нод"
}

resource "openstack_networking_secgroup_rule_v2" "ingress_ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s.id
}

resource "openstack_networking_secgroup_rule_v2" "ingress_http" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 80
  port_range_max    = 80
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s.id
}

resource "openstack_networking_secgroup_rule_v2" "ingress_https" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 443
  port_range_max    = 443
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s.id
}

resource "openstack_networking_secgroup_rule_v2" "ingress_nodeport" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30000
  port_range_max    = 32767
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s.id
}
