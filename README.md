# MedPredict ML-AI

Python-дағы зерттеу ML workspace-і: қайталанатын NHANES эксперименттері, модельдерді әділ салыстыру және тексерілетін inference API.

**Бұл клиникалық диагностикалық өнім емес.** Нәтиже диагнозды, HbA1c талдауын немесе тонометр өлшемін алмастырмайды; емді өзгертуге қолданылмайды. Қазақстан үшін клиникалық валидация жоқ.

## Қазіргі модульдер




| Модуль | Нақты мақсат | Күйі |
|---|---|---|
| D1 | HbA1c кірісте жоқ кезде сол тексерудегі HbA1c ≥6.5% критерийінің ықтималдығы | Жаңадан орындалған зерттеу; pipeline, кандидат bundle, Python API және тесттер |
| M2/SBP | Бес көрсеткіштен қазіргі систолалық қысымды бағалау | Бұрынғы эксперименттің коды мен агрегатталған дәлелі сақталған; қайта оқытылған жоқ |
| Confirmed diagnosis / басқа аурулар | Тиісті reference-standard диагноздары бар дерек қажет | Іске асырылмаған, дерек/клиникалық review қажет |

D1: жас, жыныс, BMI, жалпы холестерин және ашқарындағы глюкоза. HbA1c label ғана; input болмайды. Анамнез, дәрі және pregnancy контексті eligibility үшін тексеріледі. Бір өлшеудің label-ы расталған диабет те, диабеттің түрі де, болашақ қауіп те емес.

## Нақты алынған нәтиже

2026-10-26/27 орындалған D1 эксперименті: **2 027 қатысушы, 49 positive**. Train 1 215/29 positive; calibration 203/5; threshold 203/5; test 406/10. CV: 4 linear, 4 HGB және 3 MLP конфигурациясы; MLP әрқайсысы үш seed-та. Таңдау final test ашылғанға дейін бекітілді.

| Final internal test | Average precision | AUROC | Brier |
|---|---:|---:|---:|
| Таңдалған Logistic Regression + sigmoid calibration | 0.5205 | 0.9439 | 0.01612 |
| Тек глюкоза қолданатын Logistic Regression | **0.5547** | **0.9509** | **0.01599** |
| Train prevalence dummy | 0.0246 | 0.5000 | 0.02402 |

**Толық панельдің артықшылығы көрсетілген жоқ.** Модель test нәтижесіне қарап ауыстырылған жоқ; нейросеть CV-де жеңген жоқ. Calibration және final test positive count аз. Female test тобында positive 0 болғандықтан оның AUROC/AP көрсеткіштері анықталмайды. Бұл модель клиникалық қолдануға promotion алған жоқ; bundle — тек тексерілетін research candidate.

[D1 model card](docs/d1-model-card.md) · [орындалу есебі](docs/implementation-status-2026-10-27.md) · [машиналық test report](reports/d1/test_evaluation.json) · [data audit](reports/d1/cohort_audit.json)

## Орнату

Тексерілген орта: macOS arm64, Python 3.12.14, scikit-learn 1.7.2, PyTorch 2.9.1. Жаңа ортада:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -r requirements-serve.lock
python -m pip install --no-deps -e .
```

ML және serving dependency lock-тары бөлек. Осы жұмыстың нақты тексеруі қолданыстағы pinned ML ортасында және бөлек орнатылған serving overlay-де орындалды; таза машинадағы install әлі орындалған жоқ. Жарияланған joblib bundle тек сенімді local path-тан жүктеледі, package build/version сәйкес келмесе readiness жабылады. CI CPU Torch build-і нұсқа suffix-імен ерекшеленуі мүмкін; CI synthetic fixtures-ті тексереді, ол кандидат bundle-дің басқа ортаға жарамдылығын дәлелдемейді.

## D1 экспериментін қайталау

Ресми CDC public-use файлы ғана жүктеледі. Raw/processed жолдар, participant IDs, split manifests және жеке predictions Git-ке кірмейді.

```bash
python -m medpredict_ml.classification download
python -m medpredict_ml.classification prepare --run runs/d1-reproduction
python -m medpredict_ml.classification train --run runs/d1-reproduction
python -m medpredict_ml.classification calibrate --run runs/d1-reproduction
python -m medpredict_ml.classification evaluate --run runs/d1-reproduction
python -m medpredict_ml.classification export --run runs/d1-reproduction --bundle runs/d1-reproduction/bundle
python -m medpredict_ml.classification verify --bundle runs/d1-reproduction/bundle
```

Default config: [configs/d1_hba1c_screen.json](configs/d1_hba1c_screen.json). Final test ашылған run қайта train/calibrate/evaluate жасатпайды. Жаңа гипотезаға жаңа protocol және run керек; бір data split-ті қайта жасау тәуелсіз external validation емес. Source/config changes сақталған протоколға сай болмаса, ескі run қабылданбайды.

## Python API

Қалыпты іске қосуда модель қолжетімсіз: demo fallback жоқ. Кандидат bundle-ді explicit таңдаған local research run:

```bash
export MEDPREDICT_D1_BUNDLE="$PWD/artifacts/d1"
python -m uvicorn medpredict_ml.serving:app --host 127.0.0.1 --port 8090 --no-access-log
```

- `GET /health`: процесс тірі.
- `GET /ready`: bundle/schema/checksum/version/golden parity тексерілгенде ғана ready.
- `GET /api/v2/assessments/schema`: input JSON Schema, бірліктер және observed train support.
- `POST /api/v2/assessments`: `assessed`, `insufficient_data`, `not_applicable` немесе қауіпсіз error.
- `GET /docs`: endpoint құжаттамасы. Assessment-тің толық body schema-сы metadata endpoint-та беріледі.

[Синтетикалық request fixture](examples/d1-request.json):

```bash
curl http://127.0.0.1:8090/api/v2/assessments \
  -H 'Content-Type: application/json' \
  --data-binary @examples/d1-request.json
```

[Контракт және қосу нұсқаулығы](docs/d1-api-contract.md). Frontend және Java backend **өзгертілген жоқ**; бар v1 UI бұл жаңа response-ты өздігінен қолдамайды. Қоғамдық deployment жасалған жоқ.

## Тексеру

```bash
python -m pytest --cov=medpredict_ml --cov-report=term-missing
python tools/verify_runtime.py
```

Runtime тексеруі уақытша loopback server іске қосып, 30 synthetic request-пен parity/error/latency тексереді және server-ді тоқтатады. [Сақталған нәтиже](reports/d1/runtime_verification.json): 30/30 сәтті, probability айырмасы 0, жергілікті warm p95 шамамен 2.46 ms. Бұл SLA, browser тексеруі не clinical quality емес.

Жаңа workspace-та сол протокол қайта орындалып, selection, толық evaluation JSON және synthetic predictions дәл сәйкес шықты: [reproduction evidence](reports/d1/reproduction.json). Бұл басқа инженердің не жаңа ортаның тәуелсіз валидациясы емес.

## Құрылым және сақталған M2

- `src/medpredict_ml/classification/`: D1 data/cohort/split, CV, calibration, final evaluation және bundle.
- `src/medpredict_ml/serving.py`: stateless Python API; CORS/auth/history/storage қосылмаған.
- `configs/`, `tests/`, `tools/`: pinned experiment config, software fixtures және runtime verification.
- `reports/d1/`, `artifacts/d1/`: жаңа D1 aggregate evidence және research candidate.
- Қалған `src/medpredict_ml/*.py`, `reports/*.json`, `artifacts/*`, [M2 model card](docs/model-card.md): сақталған тарихи M2 эксперименті. Оның Java verification жазбасы осы жұмыста қайта орындалған жоқ.

M2 exact reproduction жаңа output-та орындалады: `python -m medpredict_ml <download|prepare|train|evaluate|export|verify> --workspace <new-directory>`. Root-тағы viewed-test guard тарихи нәтижені қайта tuning жасаудан қорғайды. M2-нің high-SBP қатесі және басқа шектеулері сақталған.

[Толық инженерлік жоспар](docs/planning/diagnostic-ml-plan-2026-10-26-kk.md) · [skills аудиті](docs/planning/ml-agent-skills-2026-10-26.md) · [агент нұсқаулығы](AGENTS.md)

Дерек: [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020), [NCHS terms](https://www.cdc.gov/nchs/policy/data-user-agreement.html). Third-party skills өз лицензиялары және source hashes-пен сақталған. Дерек шарттары кітапханалардың лицензиясымен ауыстырылмайды.
