"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel, MultiNMTModel

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", "MultiNMTModel", "check_sru_requirement"]
