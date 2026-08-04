SEARCH_SPACE = {
    "learning_rate": {
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    },

    "weight_decay": {
        "low": 1e-5,
        "high": 1e-1,
        "log": True,
    },

    "batch_size": [16, 32, 64],

    "optimizer": [
        "AdamW",
        "SGD",
    ]
}