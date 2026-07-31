#!/bin/bash

# Params:
# $1 = path of ud2 binary
# $2 = path of this case directory

BIN=$1
CASE_DIR=$2

$BIN ud2.yaml
if [ $? -ne 0 ]; then
    echo "ud2 binary failed"
    exit 1
fi

python3 "$CASE_DIR/validate.py" --output 5S8I_unidock2_1.json \
    --ref "$CASE_DIR/../../data/5S8I/5S8I_ligand.sdf" \
    --rmsd-limit 2.0
EXIT_CODE=$?

exit $EXIT_CODE
