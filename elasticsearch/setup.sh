docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.3
echo "IMAGE PULLED"
docker run --name elasticsearch_multi_index -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -d docker.elastic.co/elasticsearch/elasticsearch:7.9.3 
echo "SETTING UP ELASTICSEARCH CONTAINER"
sleep 60
echo "CONTAINER IS UP"
python3 elasticsearch_setup.py
echo "DONE"