# Модуль MKS (Managed Kubernetes Service)

resource "selectel_mks_cluster_v1" "main" {
  name               = "credit-scoring-${var.environment}"
  project_id         = var.project_id
  region             = var.region
  kube_version       = var.k8s_version
  enable_autorepair  = true
  network_id         = var.network_id
  subnet_id          = var.subnet_id
  maintenance_window_start = "03:00:00"
}

resource "selectel_mks_nodegroup_v1" "groups" {
  for_each = var.node_groups
  
  cluster_id        = selectel_mks_cluster_v1.main.id
  project_id        = var.project_id
  region            = var.region
  availability_zone = "${var.region}a"
  
  nodes_count   = each.value.count
  flavor_id     = data.selectel_mks_flavor_v1.flavors[each.key].id
  volume_gb     = each.value.volume_size
  volume_type   = "fast"
  
  labels = each.value.labels
  
  taints {
    key    = "workload"
    value  = each.value.labels.workload
    effect = "PreferNoSchedule"
  }
}

data "selectel_mks_flavor_v1" "flavors" {
  for_each = var.node_groups
  
  project_id = var.project_id
  region     = var.region
  filter {
    name = each.value.flavor
  }
}

# сетевые политики (базовые)
resource "selectel_mks_feature_gates_v1" "features" {
  cluster_id = selectel_mks_cluster_v1.main.id
  project_id = var.project_id
  region     = var.region
  
  feature_gates = [
    "PodSecurity"
  ]
}
