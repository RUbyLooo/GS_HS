PYTHONPATH=. python lerobot/scripts/train.py \
--policy.type=act \
--dataset.repo_id=cfy/vla_test \
--batch_size=4 \
--steps=300000 \
--output_dir=/home/ubuntu/Documents/checkpoint_il/act_ckpt \
--dataset.root=/home/ubuntu/Documents/record_data_il/single_vla_test_withoutbanana

----------------------------------------------------------------------------------
PYTHONPATH=. python lerobot/scripts/train.py \
--policy.type=act \
--dataset.repo_id=cfy/vla_test \
--batch_size=8 \
--steps=500000 \
--output_dir=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd_gripperenhance_500000 \
--dataset.root=/home/ubuntu/Documents/record_data_il/vla_epi200_withoutbanana_gripperenhance


----------------------------------------------------------------------------------
