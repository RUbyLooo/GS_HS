-------------------fixed theta, rho uniform sample-------------------
python -m lerobot.record \
--robot.type=sim_single_piper \
--robot.port=" " \
--robot.id=single_piper_robot \
--dataset.repo_id=cfy/single_piper \
--dataset.root=/home/ubuntu/Documents/record_data_il/act_epi200_fixedtheta_rhouniformsample \
--dataset.num_episodes=200 \
--dataset.single_task="Grab the cube" \
--display_data=True