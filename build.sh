#!/bin/bash

set -e

buildscripts/build_container.bash -t "multilingual:${IMAGE_TAG:-latest}" -f Containerfile .