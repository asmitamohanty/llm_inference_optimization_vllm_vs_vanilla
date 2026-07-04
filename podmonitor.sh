#!/bin/bash
set -e

source ./scripts/env.sh

echo "Run pod monitoring..."

kubectl apply -f ./scripts/podmonitoring.yaml

echo "Pod monitoring running."