# Contains the default parameters for the optimizer

lbfgsConfigDefaults = {
    "max_eval": 15,
    "max_iter": 10,
    "line_search_fn": "strong_wolfe",
    "history_size": 100,
    "tolerance_grad": 1e-8,
    "tolerance_change": 1e-6,
}
