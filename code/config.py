config = {
    "dataset":{
        "train":250,
        "dev":250,
        "test":250,
    },
    "model":{
        "learn_pos_emb":True,
        "tied_weights":False,
        "embedding_dim":64,
        "transformer_dim":64,
        "transformer_hidden_dim":128,
        "head_dim":32,
        "num_head":2,
        "num_layers":2,
        "vocab_size":512,
        "max_seq_len":2048,
        "dropout_prob":0.5, # was 0.1
        "attention_dropout":0.1,
        "pooling_mode":"MEAN",
        "num_classes": 2,
        "mixed_precision": True,
    },
    "training":{
        "batch_size":16,
        "learning_rate":0.001, # Bojian & Towards have 0.001
        "lr_decay":"linear",
        "weight_decay":0, # is this decay factor? 
        "eval_frequency": 50, # was 500
        "num_train_steps": 10, # was 20000
    }
}