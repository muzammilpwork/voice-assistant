#!/bin/bash

echo "Setting unix time as env varaible"
export UNIX_TIME=$(date +%s)

echo "Building Docker images with docker-compose"
docker-compose -f docker-compose-ci.yml build --no-cache