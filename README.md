# Divot Detect
Autonomous crater mapping on Mars

## Scope and Goal
The aim is to create and apply an ML model capable of detecting and outlining craters on Mars. Crater mapping is important for a number of reasons including: 
1) mapping terrain for eventual human exploration
2) dating geological features, 
3) acting as a way to geolocate on a planet (since we don't yet have GPS-providing satellites), and 
4) monitoring for new craters. Most places on Mars [suspected of having liquid water](https://en.wikipedia.org/wiki/Seasonal_flows_on_warm_Martian_slopes) occur on crater walls or gully walls.

After training a model to find craters in imagery, the goal is to apply that model to both Mars and the Moon. Once mapping is complete, these results will be uploaded into Arizona State University's [JMARS](https://jmars.asu.edu/) platform for public use.

The rest of this README describes how to train and deploy ML models for crater mapping. It assumes reasonable experience with ML and cloud infrastructure tools (including Google Cloud Platform, Kubernetes, and Kubeflow).

## Model Training
This repo relies on the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to train models. TF's OD API allows relatively quick training of object detection and instance segmentation models. It supports a [wide variety](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) of architectures.

Our data comes from the Mars Reconnaissance Orbiter's (MRO) as well as the Lunar Reconnaissance Orbiter Camera (LROC) and is best obtained through NASA's [PDS Image Atlas](https://pds-imaging.jpl.nasa.gov/search/). The [DREAMS lab](https://web.asu.edu/jdas/home) led by Prof. Jnaneshwar "JD" Das is spearheading the labeling effort using their [DeepGIS](https://github.com/DREAMS-lab/deepgis) tool.

Model training relies on the [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) tool.

## Model Deployment/Inference
After training, the model will detect craters for both Martian and Lunar imagery.

# Technical Training Notes

_The below walks through model training on Kubeflow. See `divdet/docs/STTR_TrainingSoftwareGuide.pdf` for more details and tips._

## Build OD API training Docker image
Before running the below, update paths in the `build_image.sh` shell script to match your desired image file location. Also update `od_export/component.yaml` and `od_train/component.yaml` to match the location of this image.
```bash
export BUILDS_DIR=<path_containing_divdet_directory>
./$BUILDS_DIR/divdet/kfpipelines/components/od_train/build_image.sh
```

## Setup Kubeflow 
Full instructions for setting up the Kubeflow cluster can be found within Kubeflow's documentation [here](https://www.kubeflow.org/docs/gke/deploy/). The two more important steps are described in detail below.

### Create management cluster
The management cluster helps setup instances of Kubeflow. Full setup instructions [here](https://www.kubeflow.org/docs/gke/deploy/management-setup/).

Example values for management cluster Makefile:
```

	kpt cfg set ./instance mgmt-ctxt management-cluster

	kpt cfg set ./upstream/manifests/gcp name kubeflow-app-v5
	kpt cfg set ./upstream/manifests/gcp gcloud.core.project divot-detect
	kpt cfg set ./upstream/manifests/gcp gcloud.compute.zone us-east1-c
	kpt cfg set ./upstream/manifests/gcp location us-east1-c
	kpt cfg set ./upstream/manifests/gcp log-firewalls false

	kpt cfg set ./upstream/manifests/stacks/gcp name kubeflow-app-v5
	kpt cfg set ./upstream/manifests/stacks/gcp gcloud.core.project divot-detect

	kpt cfg set ./instance name kubeflow-app-v5
	kpt cfg set ./instance location us-east1-c
	kpt cfg set ./instance gcloud.core.project divot-detect
	kpt cfg set ./instance email wronk@developmentseed.org
```

### Create management cluster
To deploy an instance of Kubeflow, see the guide [here](https://www.kubeflow.org/docs/gke/deploy/deploy-cli/).

```
export MGMTCTXT=management-cluster
export PROJECT=divot-detect
export KF_NAME=kubeflow-app-v5
export ZONE=us-east1-c

kubectl config use-context ${MGMTCTXT}
kubectl create namespace ${PROJECT}
kubectl config set-context --current --namespace ${PROJECT}

export CLIENT_ID=<your_client_id>
export CLIENT_SECRET=<your_client_secret>
make set-values
make apply
```

After cluster setup is complete:
```
# Verify cluster
gcloud container clusters get-credentials ${KF_NAME} --zone ${ZONE} --project ${PROJECT}
kubectl -n kubeflow get all

gcloud projects add-iam-policy-binding divot-detect --member=user:wronk@developmentseed.org --role=roles/iap.httpsResourceAccessor

# URI information
kubectl -n istio-system get ingress
export HOST=$(kubectl -n istio-system get ingress envoy-ingress -o=jsonpath={.spec.rules[0].host})

# After model training is initiated
tensorboard --logdir=gs://divot-detect/experiments/27/train
```

Add GPUs to the cluster
```
export GPU_POOL_NAME=gpu-pool
gcloud container node-pools create ${GPU_POOL_NAME} \
--accelerator type=nvidia-tesla-p100,count=1 \
--zone ${ZONE} --cluster ${KF_NAME} \
--num-nodes=1 \
--machine-type=n1-standard-16 \
--min-nodes=0 \
--max-nodes=1 \
--enable-autoscaling \
--node-taints mlUseOnly=true:NoSchedule \
--disk-size 20GB

kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## Run the TF OD API pipeline
* Open the Kubeflow UI page and navigate to the pipelines section
* Upload the pipeline .tar.gz file created after executing a python script in `/kfpipelines`

## Building Kubeflow Pipeline
```python
python divdet/kfpipelines/tf_od_pipeline_v3.py
```

## Launching Kubeflow Pipeline
See `divdet/docs/README.md` for a more detailed walkthrough on this section.

Reasonable defaults/examples for pipeline parameters:

| Parameter  | Example |
| ------------- | ------------- |
| pipeline_config_path | `gs://divot-detect/model_dev/model_templates/mask_rcnn_coco_v1.config` |
| model_dir | `gs://divot-detect/experiments/1` |
| num_train_steps | `50000` |
| sample_1_of_n_eval_examples | `10` |
| eval_checkpoint_metric | `loss` |
| metric_objective_type | `min` |
| inference_input_type | `encoded_image_string_tensor` |
| inference_output_directory | `gs://divot-detect/model_export/od1/1` |

## Clean up
```bash
kubectl delete namespace kubeflow

cd "${KF_DIR}"
make delete-gcp

# Make sure disks are deleted under Compute Engine > Disks
```


# Technical Inference Notes
_The below walks through inference using Kubernetes. See `divdet/docs/STTR_InferenceSoftwareGuide.pdf` for more details and tips._


## Create the user and database, change connection, add and verify postgis
We have used Google Cloud's SQL API to create and run a Postgres database with the PostGIS extension. However, this could be run locally or on other platforms as well. Just make sure the IP address of the database is accessible by the compute resources of your inference pipeline.

```bash
\q  # if logged in to postgres user
dropdb craters_ctx_v1  # Or `DROP DATABASE craters_ctx_v1;` if logged in
psql -d postgres;
CREATE DATABASE craters_ctx_v1 OWNER postgres;
GRANT ALL PRIVILEGES ON DATABASE craters_ctx_v1 TO postgres;

\c craters_ctx_v1 postgres
\du+

# Add and check PostGIS extension
create extension postgis;
SELECT srid, auth_name, proj4text FROM spatial_ref_sys LIMIT 10;
```

## Push messages to PubSub
```
# First modify the raw image metadata to prep for message creation
python /Users/wronk/Builds/divdet/divdet/inference/ctx_message_augment.py \
--input_data_fpath /Users/wronk/Data/divot_detect/ctx.csv \  					# Path to CTX image list containing all image metadata
--output_data_fpath /Users/wronk/Data/divot_detect/ctx_augmented.csv 			# Path to output CSV containing desired/augmented metadata

# Modify the below parameters to your liking. This will generate messages for PubSub
# See script for help messages on each CLI flag
python /Users/wronk/Builds/divdet/divdet/inference/create_messages.py \
--input_data_fpath /Users/wronk/Data/divot_detect/ctx_augmented.csv \
--output_message_fpath /Users/wronk/Data/divot_detect/ctx_messages.csv \
--csv_url_key=url \
--csv_volume_id_key=volume_id \
--csv_lon_key=center_longitude \
--csv_lat_key=center_latitude \
--csv_instrument_host_key=instrument_id \
--csv_sub_solar_azimuth_key=sub_solar_azimuth \
--prediction_endpoint=<your_prediction_endpoint_IP_address> \
--scales="[0.5, 0.25, 0.125]" \
--batch_size=2 \
--min_window_overlap=128 \
--center_reproject=False \
--latitude_threshold=15

# Push messages to PubSub
python pubsub_push_v1.py \
--input_fpath /Users/wronk/Data/divot_detect/ctx_messages.csv \
--gcp_project divot-detect \
--pubsub_topic_name divot-detect-ctx \
--pubsub_subscription_name divot-detect-ctx \
--service_account_fpath <Path_to_your_service_account>
```

## Prepare the inference pipeline parameters

First update the parameters for the following files:
* inference_gpu_pool.yaml
* inference_job_ctx.yaml
* start_ctx.sh
* service_account.yaml

Basic descriptions are in divdet/kfjobs/README.md

Then build the Docker image that will serve as a worker template:
```
docker build --no-cache -t "gcr.io/<your_project_name>/inference-pipeline-worker-ctx:v1" -f Dockerfile_inference_runner_ctx_v1 .
docker push gcr.io/<your_project_name>/inference-pipeline-worker-ctx:v1
```

## Launch the inference pipeline

```bash
cd divdet/kfjob

# Initialize the cluster with nodes and add a GPU pool K8s deployment
./cluster_initializer.sh

# modify the below Kubernetes secret, which contains DB connnection information
kubectl create secret generic k8s-db-secret-ctx-v1 \
--from-literal=username=<enter_your_DB_username> \
--from-literal=password=<enter_your_DB_username> \
--from-literal=database=<enter_your_DB_name>

# Launch the inference workers
kubectl create -f /Users/wronk/Builds/divdet/kfjob/inference_job_ctx.yaml
```

You can now view your status on the GKE landing page, watch message processing rates on PubSub, and check DB usage (if using Google's SQL API).

