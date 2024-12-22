import torch
import os
import sys
import yaml
import inspect

from model_ppca import PPCA
from model_pmf import PMF
from model_ppca_eb import PPCAEB
from model_ppca_eb_plus import PPCAEBPlus
from model_ppca_eb_column import PPCAEBColumn
from model_ppca_eb_column_plus import PPCAEBColumnPlus
from model_ppca_eb_twin import PPCAEBTwin
from model_ppca_eb_twin_plus import PPCAEBTwinPlus
from model_pmf_eb import PMFEB
from model_pmf_eb_plus import PMFEBPlus
from model_pmf_eb_column import PMFEBColumn
from model_pmf_eb_column_plus import PMFEBColumnPlus
from model_pmf_eb_twin import PMFEBTwin
from model_pmf_eb_twin_plus import PMFEBTwinPlus
from model_pmf_natural import PMFNatural
from model_pmf_natural_eb_plus import PMFNaturalEBPlus
from model_pmf_natural_eb_twin_plus import PMFNaturalEBTwinPlus
from model_pmf_natural_eb import PMFNaturalEB
from model_pmf_natural_eb_twin import PMFNaturalEBTwin
from model_ppca_natural import PPCANatural
from model_ppca_natural_eb import PPCANaturalEB
from model_ppca_natural_eb_twin import PPCANaturalEBTwin
from model_pmf_natural_eb_single import PMFNaturalEBSingle
from model_pmf_natural_eb_twin_single import PMFNaturalEBTwinSingle
from model_ppca_natural_eb_single import PPCANaturalEBSingle
from model_ppca_natural_eb_twin_single import PPCANaturalEBTwinSingle


from torch.optim import Adam, AdamW, SGD, Adadelta, NAdam, RMSprop

from torch.utils.tensorboard import SummaryWriter


# Universal
TRAINED_MODEL_TAG = "train"
RETRAINED_MODEL_TAG = "retrain"
RETRAINED_MODEL_DIR = "heldout_llhood"


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class ModelHandler:
    """Sets of utils methods to create and load models."""

    @staticmethod
    def get_model_names():
        """Get list of model names."""
        return [
            "PPCA",
            "PPCAEB",
            "PPCAEBPlus",
            "PPCAEBColumn",
            "PPCAEBColumnPlus",
            "PPCAEBTwin",
            "PPCAEBTwinPlus",
            "PPCANatural",
            "PPCANaturalEB",
            "PPCANaturalEBTwin",
            "PMF",
            "PMFEB",
            "PMFEBPlus",
            "PMFEBColumn",
            "PMFEBColumnPlus",
            "PMFEBTwin",
            "PMFEBTwinPlus",
            "PMFNatural",
            "PMFNaturalEBPlus",  
            "PMFNaturalEBTwinPlus",
            "PMFNaturalEB",
            "PMFNaturalEBTwin",
            "PMFNaturalEBSingle",
            "PMFNaturalEBTwinSingle",
            "PPCANaturalEBSingle",
            "PPCANaturalEBTwinSingle",
        ]

    @staticmethod
    def model_factory(config):
        """
        Factory function for creating the model.

        Args:
            config: A dictionary of the model configuration.

        Returns:
            A torch.nn module.
        """
        factor_model = config["factor_model"]

        if factor_model not in ModelHandler.get_model_names():
            raise ValueError(
                f"Unknown factor model {factor_model}. Have you added this model to the ModelHandler?"
            )

        new_config = config.copy()
        device = new_config["device"]
        if "summary_writer" not in new_config:
            summary_writer = SummaryWriter(new_config["param_save_dir"])
            new_config["summary_writer"] = summary_writer

        new_config["print_steps"] = (
            1 if not "print_steps" in new_config else new_config["print_steps"]
        )

        # Get the model class
        modelClass = str_to_class(factor_model)

        # Get legal arguments for the model
        legal_args = inspect.signature(modelClass.__init__).parameters.keys()

        # Remove illegal arguments
        new_config = dict(filter(lambda x: x[0] in legal_args, new_config.items()))

        # Initialize the model
        return modelClass(**new_config).to(device).double()

    @staticmethod
    def model_factory_from_path(path):
        with open(path, "r") as file:
            config = yaml.load(file, yaml.Loader)
        model = ModelHandler.model_factory(config)
        return model

    @staticmethod
    def load_model(path, retrain=False, to_train=False, device=None):
        """
        Load the model using the config dictionary.

        Args:
            path: The path to the directory that contains the model_trained.pt file
            to_train: Whether to set the model to train mode (Default: False). Will call model.train() if True or mdoel.eval() if False.

        Returns:
            The torch.nn module.
            retrain: Whether to fetch the retained (i.e., with embeddings for the holdout data), or the original model.

        Note:
            This will instantiate the model and then fill in the parameters uisng state_dict.
            This will NOT refill custom parameters or the optimizer.
        """
        train_tag = RETRAINED_MODEL_TAG if retrain else TRAINED_MODEL_TAG
        path = os.path.join(path, RETRAINED_MODEL_DIR) if retrain else path
        print(f"Loading model from {path}/model_trained_{train_tag}.pt")
        with open(os.path.join(path, "config.yaml"), "r") as file:
            config = yaml.load(file, yaml.Loader)
        if device is not None:
            config["device"] = device
        model = ModelHandler.model_factory(config)

        the_device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        if device is not None:
            the_device = torch.device(device)

        model.load_state_dict(
            torch.load(os.path.join(path, f"model_trained_{train_tag}.pt"), map_location=the_device)
        )
        model.eval() if not to_train else model.train()
        return model
