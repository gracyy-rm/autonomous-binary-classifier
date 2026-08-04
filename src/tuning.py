import optuna
import copy
from src.train import run_pipeline
from config.search_space import SEARCH_SPACE
class HyperparameterTuner:
    """
    Handles hyperparameter tuning using Optuna.
    """
    def __init__(self, config, n_trials=10):
        self.base_config = config
        self.n_trials = n_trials


    def objective(self, trial):
        """
        Objective function executed once per Optuna trial.
        """
        trial_config = copy.deepcopy(self.base_config)

        # Sample hyperparameters from the search space
        lr= trial.suggest_float(
            "learning_rate", 
            SEARCH_SPACE['learning_rate']['low'], 
            SEARCH_SPACE['learning_rate']['high'], 
            log=SEARCH_SPACE['learning_rate']['log']
        )
        
        trial_config["training_parameters"]["learning_rate"] = lr

        results = run_pipeline(config=trial_config)

        return results["best_val_acc"] 
    def optimize(self):
        """
        Run the Optuna optimization process.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print("\n============Optimization Complete============")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Value (Validation Accuracy): {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        return study
        
