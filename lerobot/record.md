python -m lerobot.record \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=single_piper_robot \
--dataset.repo_id=cfy/single_piper \
--dataset.root=/home/ubuntu/Documents/record_data_il/vla_epi20_withoutbanana \
--dataset.num_episodes=200 \
--dataset.single_task="Grab the cube" \
--display_data=True


-------------------抓取关键阶段数据增强-------------------
python -m lerobot.record \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=single_piper_robot \
--dataset.repo_id=cfy/single_piper \
--dataset.root=/home/ubuntu/Documents/record_data_il/vla_epi200_withoutbanana_dataenhance \
--dataset.num_episodes=200 \
--dataset.single_task="Grab the cube" \
--display_data=True