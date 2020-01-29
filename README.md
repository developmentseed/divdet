# Divot Detect
Autonomous crater mapping on Mars

## Scope and Goal
The aim is to create and apply an ML model capable of detecting and outlining craters on Mars. Crater mapping is important for a number of reasons including: 
1) mapping terrain for eventual human exploration
2) dating geological features, 
3) acting as a way to geolocate on a planet (since Mars doesn't have GPS satellites), and 
4) looking for liquid water in newly exposed craters. Most places on Mars [suspected of having liquid water](https://en.wikipedia.org/wiki/Seasonal_flows_on_warm_Martian_slopes) occur on crater walls or gully walls.

After training a model to find craters in imagery, the goal is to apply that model to both Mars and the Moon. After mapping craters, these results will be uploaded into Arizona State University's [JMARS](https://jmars.asu.edu/) platform for public use.

The rest of this README describes how to train and deploy ML models for crater mapping. It assumes reasonable experience with ML and cloud infrastructure tools. 

## Model Training
This repo relies on the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to train models. TF's OD API allows relatively quick training of object detection and instance segmentation models. It supports a [wide variety](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) of architectures. Limitations include the fact that it's still limited to TF v1.x (as of Jan 2020) and doesn't support multi-GPU training.

Our data comes from the Mars Reconnaissance Orbiter's (MRO) Context camera (CTX) as well as the Lunar Reconnaissance Orbiter Camera (LROC) and is best obtained through NASA's [PDS Image Atlas](https://pds-imaging.jpl.nasa.gov/search/). The [DREAMS lab](https://web.asu.edu/jdas/home) led by Prof. Jnaneshwar "JD" Das is spearheading the labeling effort using their [DeepGIS](https://github.com/DREAMS-lab/deepgis) tool.

Model training relies on the [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) tool.

## Model Deployment/Inference
After training, the model will detect craters for both Martian and Lunar imagery.

# Technical Training Notes

_The below walks through model training on Kubeflow._

## Build OD API training Docker image
Before running the below, update paths in the `build_image.sh` shell script to match your desired image file location. Also update `od_export/component.yaml` and `od_train/component.yaml` to match the location of this image.
```bash
export BUILDS_DIR=<path_containing_divdet_directory>
./$BUILDS_DIR/divdet/kfpipelines/components/od_train/build_image.sh
```
## Setup Kubeflow Control (`kfctl`) CLI tool

KFCtl makes it easy to deploy KubeFlow on a kubernetes cluster

```bash
# Setup as below or download directly from (1) https://github.com/kubeflow/kubeflow/releases or (2) https://github.com/kubeflow/kfctl/releases and place the binary file in ~/.kfctl
export KUBEFLOW_BRANCH=0.7.1
export KUBEFLOW_TAG=0.7.1-2-g55f9b2a
export PLATFORM=darwin  # Use `darwin` on Mac. Other option is `linux`
wget -P /tmp https://github.com/kubeflow/kubeflow/releases/download/v${KUBEFLOW_BRANCH}/kfctl_v${KUBEFLOW_TAG}_${PLATFORM}.tar.gz
mkdir ~/.kfctl
tar -xvf /tmp/kfctl_v${KUBEFLOW_TAG}_${PLATFORM}.tar.gz -C ${HOME}/.kfctl
```

## Setup Kubeflow on GCP cluster
```bash
# Add kfctl tool to PATH
export PATH=$PATH:${HOME}/.kfctl

# Set GCP creds
export CLIENT_ID=<my_client_id>
export CLIENT_SECRET=<my_client_secret>
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v0.7-branch/kfdef/kfctl_gcp_iap.0.7.1.yaml"

# Set GCP project vars
export PROJECT=divot-detect
export ZONE=us-east1-c
# KF_NAME can't be a full directory path, just a directory name
# Increment the version for each cluster you create. Otherwise, the cluster may have issues deploying if you delete the cluster storage
export KF_NAME=kubeflow-app-v19

# Set local deployment directory
export BASE_DIR=${BUILDS_DIR}/divdet
export KF_DIR=${BASE_DIR}/${KF_NAME}

# Create directory of configuration files
gcloud config set project ${PROJECT}
mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl build -V -f ${CONFIG_URI}

# Make some changes to your GCP config if necessary (here, turning down master resources. We'll create our own pre-emptible GPU pool)
# `brew install yq` if you don't have it installed
export GCP_CONFIG_FILE=${KF_DIR}/gcp_config/cluster-kubeflow.yaml
yq w -i ${GCP_CONFIG_FILE} resources[0].properties.cpu-pool-enable-autoscaling false
yq w -i ${GCP_CONFIG_FILE} resources[0].properties.gpu-pool-enable-autoscaling false
yq w -i ${GCP_CONFIG_FILE} resources[0].properties.gpu-pool-max-nodes 0
yq w -i ${GCP_CONFIG_FILE} resources[0].properties.cpu-pool-machine-type n1-standard-4
#yq w -i ${GCP_CONFIG_FILE} resources[0].properties.cpu-pool-disk-size 100GB
yq w -i ${GCP_CONFIG_FILE} resources[0].properties.autoprovisioning-config.enabled false

# Update the Google service account to have access to cloud storage
export IAM_CONFIG_FILE=${KF_DIR}/gcp_config/iam_bindings.yaml
yq w -i ${IAM_CONFIG_FILE} bindings[2].roles[3] roles/storage.admin

# Deploy to GCP
export CONFIG_FILE=${KF_DIR}/kfctl_gcp_iap.0.7.1.yaml
kfctl apply -V -f ${CONFIG_FILE}
```

## Update permissions to grant access needed for resources to function
```bash
gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:${KF_NAME}-user@${PROJECT}.iam.gserviceaccount.com \
--role roles/storage.objectAdmin \
--role roles/logging.logWriter \
--role roles/monitoring.editor

gcloud projects add-iam-policy-binding ${PROJECT} \
--member serviceAccount:${KF_NAME}-vm@${PROJECT}.iam.gserviceaccount.com \
--role roles/storage.objectAdmin \
--role roles/viewer
```

## Add a nodepool (to start instance that will actually run ML)
```bash
gcloud container node-pools create gpu-node-pool \
--cluster ${KF_NAME} \
--zone ${ZONE} \
--num-nodes 1 \
--enable-autoscaling --min-nodes=0 --max-nodes=1 \
--machine-type n1-highmem-8 \
--scopes cloud-platform --verbosity error \
--accelerator=type=nvidia-tesla-p100,count=1 \
--service-account ${KF_NAME}-user@${PROJECT}.iam.gserviceaccount.com \
--node-labels=mlUseIntended="true"
#--node-taints cloud.google.com/gke-preemptible=true:NoSchedule \
#--preemptible
```
## Verifying resources
This connects kubectl to your cluster.

```
gcloud container clusters get-credentials ${KF_NAME} --zone ${ZONE} --project ${PROJECT}
kubectl -n kubeflow get all
```

## Run the TF OD API pipeline
* Open the Kubeflow UI page and navigate to the pipelines section
* Upload the pipeline .tar.gz file created after executing a python script in `/kfpipelines`

## Building Kubeflow Pipeline
```python
python divdet/kfpipelines/tf_od_pipeline_v1.py
```

## Launching Kubeflow Pipeline
See `divdet/docs/README.md` for a more detailed walkthrough on this section.

Then open up the Kubeflow UI by looking in GKE > Services & Ingress. It should be something like `https://kubeflow-app-vX.endpoints.divot-detect.cloud.goog/`. Note that the  take 5-10 minutes to come online _after_ the cluster is completely ready (so be patient if you're getting `This site can’t be reached` or `This site can’t provide a secure connection` messages). Go to the Pipelines tab and upload the .tar.gz file to the Kubeflow Pipelines UI and kick off a run. 

Reasonable defaults/examples for pipeline parameters:

| Parameter  | Example |
| ------------- | ------------- |
| pipeline_config_path | `gs://divot-detect/model_dev/model_templates/mask_rcnn_coco_v1.config` |
| model_dir | `gs://divot-detect/experiments/1` |
| num_train_steps | `200000` |
| sample_1_of_n_eval_examples | `10` |
| eval_checkpoint_metric | `loss` |
| metric_objective_type | `min` |
| inference_input_type | `encoded_image_string_tensor` |
| inference_output_directory | `gs://divot-detect/model_export/od1/001` |

## Clean up
```bash
# Delete cluster/resources once finished. You can skip deleting the storage if you want to rerun the same cluster later
kfctl delete -f ${CONFIG_FILE} --delete_storage
```

# Technical Deployment Notes

_To be updated as code is developed._
