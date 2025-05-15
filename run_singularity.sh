#!/bin/bash

#module load singularity

code_dir=/home/u00483/repos/TourismDialogue

#singularity shell --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif

singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir}/container --bind ${code_dir}:/tmp/code ${code_dir}/tourism_project.sif bash -c "cd browser && export FLASK_APP=app && export PATH=${PATH}:~/.local/bin && flask run --port 9300"
