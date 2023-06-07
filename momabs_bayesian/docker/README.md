### Check server status

- images that exist on server: 'docker images'
- cpu usage: 'htop'

### build docker image

docker build -t luisa/guts .

- luisa/guts is the tag of your docker image - use this in your run.sh configuration

### open image to test stuff

docker run -v $(pwd)/:/src -it luisa/guts bash

### run job

thomcode/docker/run.sh python /src/thomcode/run_guts.py --num_objectives=2 --poly_degree=3 --sigth=0.01 --num_iter=5000
