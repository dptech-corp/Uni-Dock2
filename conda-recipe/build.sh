#!/bin/bash

set -ex

cd unidock/unidock_engine
$PYTHON -m pip install .
cd ../../

$PYTHON -m pip install .
