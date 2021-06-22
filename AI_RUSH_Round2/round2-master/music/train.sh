#!/usr/bin/env bash
TASK=q3
DATASET=rush6-1
nsml run -g 1 -c 2 -d ${DATASET} -v -a "--config_file ${TASK}/config.yaml"
