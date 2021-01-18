## List of files
* cluster_initializer.sh: Helper shell script to launch a kubernetes cluster on GCP. Modify to set the number of desired instances and make sure to add a secret specifying the database connection details
* inference_gpu_pool.yaml: Kubernetes deployment specifying the TFServing Docker image that should be used for each GPU. Also contains a load  balancer IP address where prediction requests should be sent by the Kubernetes Job workers
* inference_job_*.yaml: Kubernetes Job specifying details about each worker in the queue
* start_*.sh: Shell script initiated upon container initialization of a worker
* service_account.yaml: Kubernetes Service Account used to link the Kubernetes permissions with a Google Service Account
* ../Dockerfile_inference_runner_ctx_v1: Docker build scripts specifying the worker image.