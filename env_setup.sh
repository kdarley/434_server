sudo apt-get update -y
sudo apt-get install -y docker.io

sudo systemctl start docker
sudo systemctl enable docker

sudo usermod -aG docker $USER
newgrp docker

cd /home/ubuntu/server

docker build -t server .

docker run -d -p 8080:8080 server