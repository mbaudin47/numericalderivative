#!/bin/sh

set -xe

echo `pwd`

export PYTHONPATH="$PWD:$PYTHONPATH"

# Run the tests
cd tests
bash run-all.sh
cd ..

# Run the examples
cd examples
bash run_examples.sh
cd ..
