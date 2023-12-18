git clone https://github.com/allenai/scirepeval.git
pip install -r scirepeval/requirements.txt
cd /content/scirepeval/training
python pl_training.py --gpu 8 --batch-size 32 --tasks-config tasks_config_nonhard.json allenai/specter2_base experiment1
