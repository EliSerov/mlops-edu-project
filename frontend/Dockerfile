# Простой nginx frontend
# В реальном проекте была бы React/Vue сборка, здесь просто статичная демо страница

FROM node:20-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production || npm init -y
COPY . .
# если есть шаг сборки
RUN npm run build 2>/dev/null || true


FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html 2>/dev/null || true
COPY index.html /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
