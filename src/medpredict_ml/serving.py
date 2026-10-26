from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from .classification.bundle import Bundle, UNITS, LIMITATIONS
from .classification.data import TASK


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Values(StrictModel):
    age: StrictInt | None = Field(default=None, ge=0)
    sex: Literal["male", "female"] | None = None
    bmi: StrictFloat | StrictInt | None = Field(default=None, gt=0)
    totalCholesterol: StrictFloat | StrictInt | None = Field(default=None, gt=0)
    glucose: StrictFloat | StrictInt | None = Field(default=None, gt=0)


class Units(StrictModel):
    age: Literal["years"] | None = None
    bmi: Literal["kg/m2"] | None = None
    totalCholesterol: Literal["mmol/L"] | None = None
    glucose: Literal["mmol/L"] | None = None


class Context(StrictModel):
    fasting: StrictBool | None = None
    sameExamination: StrictBool | None = None
    hba1cKnown: StrictBool | None = None
    knownDiabetes: StrictBool | None = None
    glucoseLoweringMedication: StrictBool | None = None
    pregnancyStatus: Literal["not_pregnant", "pregnant", "unknown", "not_applicable"] | None = None


class Assessment(StrictModel):
    task: Literal["D1-HBA1C-SCREEN-v1"]
    schemaVersion: Literal["2.0"]
    values: Values
    units: Units
    context: Context


class Finding(StrictModel):
    condition: Literal["single_hba1c_ge_6_5_percent"]
    probability: float = Field(ge=0, le=1)


class Result(StrictModel):
    status: Literal["assessed", "insufficient_data", "not_applicable"]
    task: Literal["D1-HBA1C-SCREEN-v1"] = TASK
    schemaVersion: Literal["2.0"] = "2.0"
    modelVersion: str
    researchOnly: Literal[True] = True
    clinicalValidation: Literal["not_established"] = "not_established"
    findings: list[Finding]
    reasons: list[str]
    limitations: list[str] = LIMITATIONS


def create_app(bundle_path: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        if bundle_path is not None:
            try:
                app.state.bundle = Bundle(bundle_path)
            except Exception:
                pass  # Public readiness exposes availability only, never artifact paths or exceptions.
        yield

    app = FastAPI(title="MedPredict D1 research API", version="2.0", lifespan=lifespan)

    @app.middleware("http")
    async def boundary(request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse({"error": "SYSTEM_ERROR"}, status_code=500)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Pydantic errors include submitted input; do not echo values, unknown keys or diagnostics.
        return JSONResponse({"error": "INVALID_REQUEST", "message": "Invalid assessment structure, type or unit"}, status_code=400)

    def available():
        return getattr(app.state, "bundle", None)

    @app.get("/health")
    def health():
        return {"status": "UP"}

    @app.get("/ready")
    def ready():
        bundle = available()
        if bundle is None:
            return JSONResponse({"error": "MODEL_UNAVAILABLE"}, status_code=503)
        return {"status": "READY", "task": TASK, "modelVersion": bundle.metadata["modelVersion"], "researchOnly": True}

    @app.get("/api/v2/assessments/schema")
    def schema():
        bundle = available()
        if bundle is None:
            return JSONResponse({"error": "MODEL_UNAVAILABLE"}, status_code=503)
        return {"task": TASK, "schemaVersion": "2.0", "researchOnly": True, "ready": True,
            "modelVersion": bundle.metadata["modelVersion"], "featureOrder": bundle.metadata["featureOrder"],
            "units": UNITS, "support": bundle.metadata["support"], "ageBySex": bundle.metadata["ageBySex"],
            "requestSchema": Assessment.model_json_schema(), "limitations": LIMITATIONS}

    @app.post("/api/v2/assessments", response_model=Result)
    async def assess(request: Request):
        # Decode once with duplicate-key rejection; the usual JSON decoder silently overwrites keys.
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate key")
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError("Nonfinite JSON constant")
        try:
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 16_384:
                    raise ValueError("Request too large")
                body.extend(chunk)
            if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                raise ValueError("Invalid request")
            raw = json.loads(body, object_pairs_hook=unique, parse_constant=invalid_constant)
            payload = Assessment.model_validate(raw)
        except (ValueError, TypeError):
            return JSONResponse({"error": "INVALID_REQUEST", "message": "Invalid assessment structure, type or unit"}, status_code=400)
        bundle = available()
        if bundle is None:
            return JSONResponse({"error": "MODEL_UNAVAILABLE"}, status_code=503)
        values, units, context = payload.values.model_dump(), payload.units.model_dump(), payload.context.model_dump()
        version = bundle.metadata["modelVersion"]
        missing = ["values."+k for k, v in values.items() if v is None]
        missing += ["units."+k for k, v in units.items() if v is None]
        missing += ["context."+k for k, v in context.items() if v is None or (k == "pregnancyStatus" and v == "unknown")]
        if missing:
            return Result(status="insufficient_data", modelVersion=version, findings=[], reasons=missing)
        reasons = []
        for key, expected in {"fasting": True, "sameExamination": True, "hba1cKnown": False,
                              "knownDiabetes": False, "glucoseLoweringMedication": False}.items():
            if context[key] != expected:
                reasons.append("context."+key)
        sex = values["sex"]
        if context["pregnancyStatus"] != ("not_pregnant" if sex == "female" else "not_applicable"):
            reasons.append("context.pregnancyStatus")
        vector = [values[k] if k != "sex" else {"male": 0, "female": 1}[sex] for k in bundle.metadata["featureOrder"]]
        for key, value in zip(bundle.metadata["featureOrder"], vector):
            lo, hi = bundle.metadata["support"][key]
            if not lo <= value <= hi:
                reasons.append("support."+key)
        lo, hi = bundle.metadata["ageBySex"].get(sex, [1, 0])
        if not lo <= values["age"] <= hi:
            reasons.append("support.ageBySex")
        if reasons:
            return Result(status="not_applicable", modelVersion=version, findings=[], reasons=reasons)
        try:
            probability = float(bundle.probabilities([vector])[0])
            return Result(status="assessed", modelVersion=version, findings=[Finding(
                condition="single_hba1c_ge_6_5_percent", probability=probability)], reasons=[])
        except Exception:
            return JSONResponse({"error": "MODEL_UNAVAILABLE"}, status_code=503)

    return app


app = create_app(Path(os.environ["MEDPREDICT_D1_BUNDLE"]) if os.environ.get("MEDPREDICT_D1_BUNDLE") else None)
