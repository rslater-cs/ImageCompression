rm -r miniconda3/bin/conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 -s
rm Miniconda3-latest-Linux-x86_64.sh
conda=miniconda3/bin/conda
$conda env create -f '/user/HS225/rs01241/Documents/disso/ImageCompression/enviroment/ubuntu_env.yml'