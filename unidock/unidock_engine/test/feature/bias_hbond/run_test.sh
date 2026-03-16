#!/bin/bash

# Params:
# $1 = path of ud2 binary
# $2 = path of cases

BIN=$1

# perform ud2 binary

$BIN ud2.yaml

# check the output
python3 validate.py
EXIT_CODE=$?

# 0=success
exit $EXIT_CODE




