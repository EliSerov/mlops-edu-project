# Простой nginx frontend
# В реальном проекте была бы React/Vue сборка, здесь просто статичная демо страница

FROM nginx:alpine

COPY index.html /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
