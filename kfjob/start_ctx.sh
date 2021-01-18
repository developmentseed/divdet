#!/bin/bash
python3 divdet/inference/inference_runner.py \
--gcp_project divot-detect \
--pubsub_subscription_name divot-detect-ctx \
--database_uri postgres+pg8000://postgres:postgres@127.0.0.1:5432/craters_ctx_v1