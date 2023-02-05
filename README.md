# mnist_classification

The idea of this repo is to do a quick and easy build of a Docker container to perform online inference with trained machine learning models which in this case is pytorch image classification model using Python APIs with Flask.

#To run the dockerfile, use this command:
docker image build -t flask_docker .

#To run the docker container:
docker run -p 5000:5000 -d flask_docker
