import torch


class Config:
    # Data parameters
    max_neighbors = 200
    neighbor_distance_threshold = 150
    biomarker_dim = 512

    # Model parameters
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # Training parameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100

    # ====== INTENSITY MASKING CONFIGURATION ======
    center_intensity_mask_probability = 0.15
    center_intensity_min_mask_ratio = 0.1
    center_intensity_max_mask_ratio = 0.8
    center_intensity_mask_value = 0.0

    neighbor_intensity_mask_probability = 0.1
    neighbor_intensity_min_mask_ratio = 0.05
    neighbor_intensity_max_mask_ratio = 0.3
    neighbor_intensity_mask_value = 0.0

    enable_center_intensity_masking = False
    enable_neighbor_intensity_masking = False

    center_mask_strategy = 'random'
    neighbor_mask_strategy = 'random'
    preserve_top_biomarkers = 0

    prediction_loss_weight = 1.0
    celltype_loss_weight = 0.0
    reconstruction_loss_weight = 0.0
    # === MODIFICATION END ===

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    data_path = r'C:\Users\alexk\PycharmProjects\Model_v1\data'
    model_save_path = './models'

    # Data loading parameters
    min_cells_per_region = 10
    min_biomarkers_per_cell = 1

    # Expected CSV filenames
    celltype_filename = 'cell_type.csv'
    position_filename = 'cell_data.csv'
    biomarker_filename = 'expression.csv'

    # ====== BIOMARKER EMBEDDING CONFIGURATION ======
    # SAFE DEFAULT: Use random embeddings (always works)
    embedding_method = 'onehot'  # Options: 'esm', 'gpt', 'random', 'onehot'
    recompute_embeddings = True  # Force recomputation to avoid cached issues

    # Method-specific parameters
    gpt_embedding_dim = 64
    random_embedding_dim = 512
    random_seed = 42

    # ESM-specific
    biomarker_dim = 512

    # Legacy parameters
    mask_probability = 0.15
    min_mask_ratio = 0.1
    max_mask_ratio = 0.8
    mask_token_value = 0.0