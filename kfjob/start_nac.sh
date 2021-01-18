#!/bin/bash
python3 divdet/inference/inference_runner.py \
--gcp_project divot-detect \
--pubsub_subscription_name divot-detect-nac \
--database_uri postgres+pg8000://postgres:postgres@127.0.0.1:5432/craters_nac_v1