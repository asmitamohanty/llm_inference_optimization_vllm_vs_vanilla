#!/bin/bash
set -e

source ./scripts/env.sh

echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

echo "Setting region..."
gcloud config set compute/region $REGION

echo "Setting zone..."
gcloud config set compute/zone $ZONE

echo "Checking auth..."
gcloud auth list

echo "Current config:"
gcloud config list

echo "Enable cloud API:"
gcloud services enable cloudbuild.googleapis.com

gcloud services list --enabled | egrep "cloudbuild|container|artifactregistry"