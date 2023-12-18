!pip uninstall torchvision -y
!pip install torchvision
!pip uninstall tensorflow protobuf -y
!pip install tensorflow
!pip install "numpy<1.25.0,>=1.18.5"
!pip install protobuf==3.20.*


!pip uninstall torchtext -y
!pip install torchtext


git clone https://github.com/allenai/scirepeval.git
pip install -r scirepeval/requirements.txt
cd /content/scirepeval/training
python pl_training.py --gpu 8 --batch-size 32 --tasks-config tasks_config_nonhard.json allenai/specter2_base experiment1
