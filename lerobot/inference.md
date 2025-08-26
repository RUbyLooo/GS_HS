python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd/checkpoints/020000/pretrained_model


python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd/checkpoints/060000/pretrained_model


python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd/checkpoints/120000/pretrained_model


-----------------------------------------有了gripper增强以后--------------------------------------
python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd_gripperenhance/best_model/pretrained_model

python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd_gripperenhance/checkpoints/060000/pretrained_model

因为有效果所以增大了训练步数到500，000，并且测试batch_size的影响

在训练到40,000步的时候batechsize的改变没有什么区别
python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint_il/act_ckpt_with3rd_gripperenhance_500000/best_model/pretrained_model

-----------------------------------------增强抓取以后--------------------------------------
python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=act \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint/ACT/debug_act_ckpt/best_model/pretrained_model

--robot.type=sim_single_piper 
--robot.port=" " 
--robot.id=sim_single_piper_robot 
--policy.type=act 
--task="Grab the cube" 
--ckpt_path=/home/ubuntu/Documents/checkpoint/ACT/debug_act_ckpt/best_model/pretrained_model

-----------------------------------------diffusion训练结果--------------------------------------
python -m lerobot.inference \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=sim_single_piper_robot \
--policy.type=diffusion \
--task="Grab the cube" \
--ckpt_path=/home/ubuntu/Documents/checkpoint/diffusion/diffusion_withoutbanana_fixtedradius_succeedstop_enhance/best_model/pretrained_model
