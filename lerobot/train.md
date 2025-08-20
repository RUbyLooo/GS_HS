PYTHONPATH=. python lerobot/scripts/train.py \
--policy.type=act \
--dataset.repo_id=cfy/vla_test \
--batch_size=4 \
--steps=1000000 \
--output_dir=/home/ubuntu/Documents/checkpoint/ACT/act_ckpt \
--dataset.root=/home/ubuntu/Documents/record_data/ACT/vla_epi200_withoutbanana_fixtedradius_succeedstop_enhance

--policy.type=act 
--dataset.repo_id=cfy/vla_test 
--batch_size=4 
--steps=1000000 
--output_dir=/home/ubuntu/Documents/checkpoint/ACT/debug_act_ckpt
--dataset.root=/home/ubuntu/Documents/record_data/ACT/vla_epi200_withoutbanana_fixtedradius_succeedstop_enhance

----------------------------------------------------------------------------------
PYTHONPATH=. python lerobot/scripts/train.py \
--policy.type=diffusion \
--dataset.repo_id=cfy/vla_test \
--batch_size=4 \
--steps=1000000 \
--output_dir=/home/ubuntu/Documents/diffusion/diffusion_withoutbanana_fixtedradius_succeedstop_enhance \
--dataset.root=/home/ubuntu/Documents/record_data/ACT/vla_epi200_withoutbanana_fixtedradius_succeedstop_enhance


----------------------------------------------------------------------------------
