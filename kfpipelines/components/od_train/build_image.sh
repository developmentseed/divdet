#!/bin/bash -e

image_name=gcr.io/divot-detect/divot-detect-training
image_tag=v4-gpu
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")" 
docker build --no-cache -t "${full_image_name}" -f Dockerfile_v2 .
docker push "$full_image_name"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"
