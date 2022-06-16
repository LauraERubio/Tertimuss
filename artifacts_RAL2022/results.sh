#!/bin/bash

source .venv/bin/activate
echo "Results from simulations"

python read_resultsCASE.py -m 2
python read_resultsCASE.py -m 4

