# S3-совместимое объектное хранилище для DVC, моделей и т.д.

resource "selectel_vpc_user_v2" "s3" {
  password = random_password.s3.result
}

resource "random_password" "s3" {
  length  = 16
  special = false
}

resource "openstack_objectstorage_container_v1" "data" {
  name = var.bucket_name
}

# креды для доступа DVC/приложения
resource "selectel_s3_credentials_v1" "main" {
  project_id = var.project_id
  user_id    = selectel_vpc_user_v2.s3.id
}
