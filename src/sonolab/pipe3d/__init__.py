"""True 3D scalar acoustic pipe survey Lab A + assembly product."""

from __future__ import annotations

from sonolab.pipe3d.acquire import acquire_pipe_survey_3d
from sonolab.pipe3d.block8 import run_block8_demo
from sonolab.pipe3d.constants import C_PROP, PHYSICS_LABEL, PIPE_R
from sonolab.pipe3d.display import build_display_wall
from sonolab.pipe3d.geometry import true_standoff
from sonolab.pipe3d.survey import (
    SurveyPack,
    SurveyTruth,
    assemble_wall,
    estimate_survey,
    score_survey,
)

__all__ = [
    "C_PROP",
    "PHYSICS_LABEL",
    "PIPE_R",
    "SurveyPack",
    "SurveyTruth",
    "acquire_pipe_survey_3d",
    "assemble_wall",
    "build_display_wall",
    "estimate_survey",
    "run_block8_demo",
    "score_survey",
    "true_standoff",
]
