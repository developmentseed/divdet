#!/bin/bash
# Convenience script to start up a kubernetes cluster on GCP

# Create CPU node pool for inference runners annd GPU node pool for prediction
gcloud container clusters create inference-cluster --num-nodes 2 --machine-type n1-highmem-8 --disk-size 40GB --enable-stackdriver-kubernetes --workload-pool=divot-detect.svc.id.goog

gcloud container node-pools create cpu-pool-preemptible --num-nodes 1 --machine-type n1-highmem-8 --disk-size 40GB --cluster inference-cluster --preemptible
#gcloud container node-pools create gpu-pool --num-nodes 1 --machine-type n1-standard-4 --disk-size 40GB --accelerator=type=nvidia-tesla-p100,count=1 --cluster inference-cluster
gcloud container node-pools create gpu-pool-preemptible --num-nodes 4 --machine-type n1-standard-4 --disk-size 40GB --accelerator=type=nvidia-tesla-p100,count=1 --cluster inference-cluster --preemptible

# Add nvidia drivers. Make sure you use a driver version compatible with other components of your code (e.g., TF)
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-nvidia-v450.yaml

# Set permissions
kubectl apply -f /Users/wronk/Builds/divdet/kfjob/service_account.yaml
gcloud iam service-accounts add-iam-policy-binding \
--role roles/iam.workloadIdentityUser \
--member "serviceAccount:divot-detect.svc.id.goog[default/ksa-cloud-sql-v1]" \  # Sub your service account here
cloud-sql-v1@divot-detect.iam.gserviceaccount.com

# Add KSA to google SA
kubectl annotate serviceaccount \
ksa-cloud-sql-v1 \
iam.gke.io/gcp-service-account=cloud-sql-v1@divot-detect.iam.gserviceaccount.com

# Start tensorflow serving images
kubectl create -f /Users/wronk/Builds/divdet/kfjob/inference_gpu_pool.yaml

# Add kubernetes secret specifying ENV vars that will be added. Make sure not to commit these
#kubectl create secret generic k8s-db-secret-v1 \
#--from-literal=username=<FILL IN USERNAME> \
#--from-literal=password=<FILL IN PASSWORD> \
#--from-literal=database=<FILL IN DATABASE NAME>
