FROM airvision_base:latest

WORKDIR /usr/srv

# # Create the ubuntu user with proper flags
# RUN useradd -m -d /code -s /bin/bash -u 1001 ubuntu

COPY . /usr/srv/user_management

RUN chmod 755 /usr/srv/user_management/entrypoint.sh
RUN chown -R ubuntu:ubuntu /usr/srv
RUN mkdir -p /usr/srv/user_management/migrations/versions \
    && chown -R ubuntu:ubuntu /usr/srv/user_management/migrations \
    && chmod -R 755 /usr/srv/user_management/migrations
# Verify that the entrypoint file exists and has correct permissions
RUN ls -la /usr/srv/user_management/entrypoint.sh

USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/user_management/entrypoint.sh"]