# Instructions to run mean Average Precision using evalzoo/matric in docker desktop/R

Follow the instructions here https://github.com/cytomining/evalzoo/blob/main/matric/README.md to install Docker and clone the Docker Image

Paste the following in a command prompt:

docker run --rm -ti -v C:/Users/Fer/Desktop/evalzoo/:/input -e PASSWORD=rstudio -p 8787:8787 shntnu/evalzoo:latest

Open http://localhost:8787/ on web browser, using User rstudio and password rstudio

Open the project evalzoo.rproj, go to the terminal and do:

git pull

Paste the following:

setwd("matric")
source("run_param.R")

Upload your yaml file to the metrics/params folder. Make sure your files are inside the input folder you determined in the first command above (docker run…). Then run the following:

run_param("params/params.yaml", results_root_dir = "/input/output/")





