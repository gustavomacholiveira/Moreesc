#!/bin/bash
# -*- coding: utf-8 -*-

python_scripts=`ls fig.*.py`

# Run these first
for script in \
    "fig.clarinetlike.reeddamping.py" \
    "fig.trumpet.impedance.py" \
    "fig.trumpet.simulation_down.py" \
    "fig.trumpet.simulation_up.py" ; do
  echo "##" $script
  python $script
done;

for script in $python_scripts; do
  echo "##" $script
  python $script
done;
