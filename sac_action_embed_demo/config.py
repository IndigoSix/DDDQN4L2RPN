class Config:
    sac_ver = 'v1'
    env_name = "l2rpn_case14_sandbox"   # Default
    # env_name = "l2rpn_neurips_2020_track2_x1"

    if env_name == "l2rpn_case14_sandbox":

        # for embedding
        state_embed_dim = 48
        action_embed_dim = 32

        state_embed_hiddens = [128, 64]
        action_embed_hiddens = [128, 64]
        ac_hiddens = [128, 64]

    elif env_name == "l2rpn_neurips_2020_track2_x1":
        # for embedding
        state_embed_dim = 128
        action_embed_dim = 128

        state_embed_hiddens = [256, 256]
        action_embed_hiddens = [256, 256]
        ac_hiddens = [256, 256]

    state_dims = None
    action_dims = None

    cell_num = 32
    seq_len = 10
    action_embed_lr = 0.001

    # for ac
    gamma = 0.99
    alpha_lr = 0.001
    actor_lr = 0.00005  # 25
    critic_lr = 0.001
    tau = 0.995
    alpha = 0.2

    summary_folder = "summary/" + env_name
    model_save_folder = "saved_models/" + env_name

    # for agent
    global_max_step = 4000000
    warm_up = 3000
    max_step = 8000
    episodes = 30000
    memory_size = 100000
    batch_size = 32
    action_batch_size = 32
    REW_ver = None