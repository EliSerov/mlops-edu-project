# Конфиг backend для Terraform state
# Использование: terraform init -backend-config=backend.hcl

bucket         = "credit-scoring-tfstate"
endpoint       = "https://s3.storage.selcloud.ru"
region         = "ru-3"
skip_credentials_validation = true
skip_metadata_api_check     = true
skip_region_validation      = true
force_path_style            = true
