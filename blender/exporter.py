import bpy
import numpy as np
from dataclasses import dataclass
from ..nitro.model import ImportedModel


@dataclass
class ExportOptions:
    pass


def export_nsbmd(context: bpy.types.Context, filepath: str, opts: ExportOptions = ExportOptions()):
    pass


def scene_to_ir(context: bpy.types.Context, opts: ExportOptions) -> ImportedModel:
    pass


def build_nsbmd(model: ImportedModel, stash: bytes | None = None) -> bytes:
    pass
