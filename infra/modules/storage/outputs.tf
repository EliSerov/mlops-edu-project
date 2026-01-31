output "bucket_name" {
  value = openstack_objectstorage_container_v1.data.name
}

output "endpoint" {
  value = "https://s3.storage.selcloud.ru"
}

output "access_key" {
  value     = selectel_s3_credentials_v1.main.access_key
  sensitive = true
}

output "secret_key" {
  value     = selectel_s3_credentials_v1.main.secret_key
  sensitive = true
}
