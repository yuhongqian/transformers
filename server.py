import logging
import time
from typing import Dict, Tuple
import os
import psutil
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    AutoModel,
)
from transformers.hf_api import HfApi, ModelInfo


_logger = logging.getLogger("transformers.configuration_utils")
_logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)


PIPELINES: Dict[str, Pipeline] = {}

MAPPING: Dict[str, Tuple[AutoModel, Pipeline]] = {
    "sequence-classification": (AutoModelForSequenceClassification, TextClassificationPipeline,),
    "token-classification": (AutoModelForTokenClassification, TokenClassificationPipeline,),
}

"""
Get all model infos from https://huggingface.co/models
"""
model_infos = [x for x in HfApi().model_list() if "pytorch" in x.tags]
print(len(model_infos))


def load_model(model_info: ModelInfo) -> None:
    """
    Method to load *one* model
    """
    model_id = model_info.modelId

    model_class = None
    pipeline_class = None
    for tag, (_model_class, _pipeline_class) in MAPPING.items():
        if tag in model_info.tags:
            model_class = _model_class
            pipeline_class = _pipeline_class

    if model_class is None:
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except OSError:
        logger.warning(f"Error loading tokenizer for: {model_id}")
        return
    try:
        model = model_class.from_pretrained(model_id)
    except OSError:
        logger.warning(f"Error loading model for: {model_id}")
        return
    pipeline = pipeline_class(model=model, tokenizer=tokenizer)
    PIPELINES[model_id] = pipeline


"""
Load all the pipelines we can.
"""
start = time.time()
for model_info in model_infos:
    mem_available = psutil.virtual_memory().available
    print("🔥", mem_available, "⚡️", len(PIPELINES))
    if mem_available > 10 ** 9:
        load_model(model_info)
    if os.environ.get("DEBUG") and len(PIPELINES) > 2:
        break
logger.info("[Took %.3f s] Loading models", time.time() - start)


async def homepage(request):
    print(request)
    pipelines_info = [(model_id, p.__class__.__name__) for (model_id, p) in PIPELINES.items()]
    return JSONResponse(pipelines_info)


async def forward(request):
    model_id = request.path_params["model_id"]
    pipeline = PIPELINES.get(model_id)
    if pipeline is None:
        return PlainTextResponse("Not Found", status_code=404)
    inputs = [
        "I love this movie.",
        "This movie sucks big time.",
        "My name is Julien and I live in Brooklyn.",
    ]
    outputs = pipeline(inputs)
    return JSONResponse(outputs)


app = Starlette(debug=True, routes=[Route("/", homepage), Route("/models/{model_id:path}", forward)])


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", workers=1)
