# Setup Elastic Search To Compare Search Results
We run elastic search in a docker container for simplicity and ease of use, as we don't need to install the actual application on the system explicitly.\
#### First time setup
Run the setup.sh file to pull the image, start a container and load it with the data present in datasets folder.\
This takes quite a bit of time as it has to create an index for each of the files in the Datasets folder. Be Patient.
#### To stop a running container
Run the command docker stop elasticsearch_multi_index.
#### To start an existing container 
Run the command docker run elasticsearch_multi_index and wait 30 seconds befor making requests.