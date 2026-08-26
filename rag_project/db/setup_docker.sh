#!/usr/bin/env bash
set -e

CONTAINER_NAME="rag_postgres"
IMAGE="pgvector/pgvector:pg16"
PORT=5432
DB_USER="postgres"
DB_PASS="postgres"
DB_NAME="rag_db"
VOLUME_NAME="rag_pgvector_data"

echo "=== Setting up PostgreSQL with pgvector container ($CONTAINER_NAME) ==="

if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Found existing container $CONTAINER_NAME."
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Starting container $CONTAINER_NAME..."
        docker start $CONTAINER_NAME
    else
        echo "Container $CONTAINER_NAME is already running."
    fi
else
    echo "Creating persistent volume $VOLUME_NAME..."
    docker volume create $VOLUME_NAME

    echo "Running new container $CONTAINER_NAME on port $PORT..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p ${PORT}:5432 \
        -e POSTGRES_USER=$DB_USER \
        -e POSTGRES_PASSWORD=$DB_PASS \
        -e POSTGRES_DB=$DB_NAME \
        -v ${VOLUME_NAME}:/var/lib/postgresql/data \
        --restart unless-stopped \
        $IMAGE
fi

echo "Waiting for PostgreSQL to be ready..."
until docker exec $CONTAINER_NAME pg_isready -U $DB_USER -d $DB_NAME; do
    echo "Waiting for database handshake..."
    sleep 1
done

echo "PostgreSQL with pgvector is ready on localhost:$PORT!"
