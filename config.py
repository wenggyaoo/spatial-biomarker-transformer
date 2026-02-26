import torch


class Config:
    device = torch.device('cuda:0')

    # ====== Model parameters ======
    d_model = 512
    nhead = 8
    num_encoder_layers = 12
    dim_feedforward = 2048
    dropout = 0.1
    positional_embedder = "rotational"

    # ====== Training parameters ======
    model_save_path = 'model_store'
    
    batch_size = 256
    ema_momentum = 0.999
    n_subgraphs_per_region = 500

    celltype_loss_weight = 0.0
    reconstruction_loss_weight = 0.8
    distillation_loss_weight = 0.2

    lr = 1e-4
    initial_lr = 1e-6
    weight_decay = 5e-4
    warmup_steps = 1000
    
    num_epochs = 100
    wandb_project = "TEST"
    wandb_name = "placeholder"

    # min_lr = 1e-7
    # T_0 = 20
    # T_mult = 2
    # ====== DATA NORMALIZATION CONFIGURATION ======
    normalize_masked_expression = False

    # ====== SAMPLER CONFIGURATION ======
    max_neighbors = 100
    neighbor_distance_threshold = 150

    # ====== INTENSITY MASKING CONFIGURATION ======
    enable_center_intensity_masking = True  # Changed from False to True
    center_intensity_mask_probability = 1
    center_intensity_min_mask_ratio = 0.8
    center_intensity_max_mask_ratio = 0.8
    center_intensity_mask_value = 0.0
    center_mask_strategy = 'random'
    preserve_top_biomarkers = 0

    enable_neighbor_intensity_masking = True  # Changed from False to True
    neighbor_intensity_mask_probability = 1
    neighbor_intensity_min_mask_ratio = 0.2
    neighbor_intensity_max_mask_ratio = 0.8
    neighbor_intensity_mask_value = 0.0
    neighbor_mask_strategy = 'random'

    # ====== DATA LOADING CONFIGURATION ======
    biomarker_sequence_file = 'biomarker_sequences.csv'
    biomarker_name_mapping_file = 'biomarker_name_mapping.json'

    # Data loading parameters
    min_cells_per_region = 100
    min_biomarkers_per_cell = 10

    use_cell_types = True

    # Expected CSV filenames
    position_filename = 'cell_data.csv'
    biomarker_filename = 'expression.csv'
    celltype_filename = 'cell_type.csv'

    # ====== BIOMARKER EMBEDDING CONFIGURATION ======
    # SAFE DEFAULT: Use random embeddings (always works)
    embedding_method = 'learnable'  # Options: 'esm', 'onehot', 'learnable'
    recompute_embeddings = False  # Force recomputation to avoid cached issues

    # Method-specific parameters
    random_seed = 42
    learnable_embedding_dim = 512  # For learnable embeddings
    gpt_embedding_dim = 64

    # # ====== CELL TYPE TASK CONFIGURATION ======
    # # Linear probing settings
    # enable_linear_probing = True  # Enable/disable linear probing task
    # freeze_backbone_for_probing = True  # True = Option B (frozen), False = Option A (multi-task)
    # linear_probe_lr = 1e-3  # Separate learning rate for linear probe
    # linear_probe_epochs = 100  # How many epochs to train the probe

    # linear_probe_scheduler_type = 'cosine'  # Options: 'cosine', 'linear', 'constant_warmup', 'reduce_on_plateau'
    # linear_probe_warmup_ratio = 0.1  # 10% of epochs for warmup (e.g., 10 epochs if total is 100)
    # linear_probe_weight_decay = 0.01  # L2 regularization for AdamW
    # linear_probe_max_grad_norm = 1.0  # Gradient clipping threshold

    # # KNN evaluation settings
    # enable_knn_evaluation = True  # Enable/disable KNN evaluation
    # knn_k_values = [1, 3, 5, 10]  # Different K values to test
    # knn_eval_frequency = 5  # Evaluate KNN every N epochs
    
    # ====== LEGACY TESTING CONFIGURATION ======
    mask_probability = None
    min_mask_ratio = None
    max_mask_ratio = None
    mask_token_value = None

    spec_index_recon = False
