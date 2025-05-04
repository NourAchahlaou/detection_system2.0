# Stage 1: Build the React app
FROM node:18 AS build

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./airbus_ui/package.json ./airbus_ui/package-lock.json ./
RUN npm install

# Copy the rest of the app
COPY ./airbus_ui .

# Build the app
RUN npm run build

# Stage 2: Serve the app with Nginx
FROM nginx:alpine

# Copy built React files to Nginx's html directory
COPY --from=build /app/build /usr/share/nginx/html

# Copy the custom nginx config
COPY ./airbus_ui/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
