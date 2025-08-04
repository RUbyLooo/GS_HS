PYTHONPATH=. python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=kelo0234/vla_test \
--batch_size=1 \
--steps=200000 \
--output_dir=/sda1/lerobot/train/GS_real/pi0 \
--dataset.root=/sda1/lerobot/data/folding_cloth