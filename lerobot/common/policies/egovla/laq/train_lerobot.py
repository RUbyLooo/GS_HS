from lerobot.common.policies.egovla.laq.laq_model import LAQTrainer
from lerobot.common.policies.egovla.laq.laq_model import LatentActionQuantization

chunksize = 50
batchsize = 1

laq = LatentActionQuantization(
    dim = 512,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
    is_use_action_state=True,
    is_use_fk_state=True,  # 是否使用 FK 状态
    state_dim=16,
    chunksize=chunksize
).cuda()


trainer = LAQTrainer(
    laq,
    folder = '/home/cfy/data/icra_data/data/fold_cloth/data/chunk-000',
    offsets = chunksize,
    batch_size = batchsize,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 1000005,
    results_folder='/home/cfy/cfy/gs_hs_backup/gs_hs/lerobot/common/policies/egovla/laq/lap_ckph',
    lr=1e-4,
    save_model_every=20000,
    save_results_every=100,
    input_mode="top_cam_state",               # 输入数据的模式
    
)

trainer.train()        

