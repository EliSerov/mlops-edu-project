# Мониторинг — в основном плейсхолдеры т.к. стек деплоится через Helm

# Этот модуль провизионит облачный мониторинг если нужно
# Реальный Prometheus/Grafana/Loki деплоится в k8s через Helm charts

locals {
  monitoring_namespace = "monitoring"
}

# плейсхолдер для интеграции с облачным мониторингом
# у selectel есть managed monitoring, но мы используем self-hosted стек

output "note" {
  value = "Деплой мониторинг стека через Helm: см. deploy/monitoring/"
}
