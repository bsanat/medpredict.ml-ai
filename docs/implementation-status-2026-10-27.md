# ML/AI жоспарының орындалу есебі

Күні: 27 қазан 2026, Asia/Almaty. Scope пайдаланушының нақтылауымен **тек ML/AI**: frontend және Java backend өзгертілмейді. GitHub нысанасы — `bsanat/medpredict.ml-ai`. Бұл есеп бастапқы жоспарды «бәрі аяқталды» деп алмастырмайды: ғылыми/клиникалық және басқа репозиторийлерге тәуелді тармақтар төменде ашық көрсетілген.

## Орындалған жұмыс

1. Бөлек бастапқыда бос Git checkout дайындалды. Қолданыстағы frontend workspace, оның uncommitted өзгерістері және M2 weights/report-тары өзгертілмеді.
2. M2 Python коды, config, тесттер, шағын артефакттар және aggregate evidence standalone пакетке сақталды. Тарихи Java/ONNX тексерулері жаңа тексеру деп көрсетілмеді.
3. D1 classifier-ге бөлек task/config/run құрылды. CDC-ден ресми P_GHB/P_DIQ алынып, қалған ресми sources-пен бірегей SEQN арқылы байланыстырылды. Source SHA-256/terms және барлық алып тастау сандары тіркелді.
4. Data audit протоколды нақтылады: DIQ medication skips — observed no-treatment емес. D1-r2 бұл айырманы split/training-ге дейін бекітті. Unknown female pregnancy status алынып тасталды; male structurally inapplicable status бөлек өңделді.
5. 60/10/10/20 stratified participant split, train-only preprocessing, CV және MLP train-internal early stopping/refit орындалды. 11 конфигурация: 4 linear, 4 HGB, 3 MLP; MLP 3 seed-та. Барлығы 51 outer-fold fit, қосымша final refit/calibration және prespecified baselines.
6. CV бойынша linear C=.01 таңдалды. Sigmoid бөлек calibration бөлігінде fit болды. Клиникалық threshold жоқ; 0.1/0.2/0.5 exploratory operating point тек зерттеу есебінде.
7. Final test бір рет ашылып, baseline, calibration, subgroup және 1000 conditional design-aware bootstrap replicate есептелді. Test-ке қарап модель/seed өзгертілген жоқ.
8. Versioned research bundle, checksum/schema/package және synthetic golden parity тексеруі іске асырылды. Default API unavailable; candidate explicit конфигурацияда ғана жүктеледі.
9. Python FastAPI сервисі: strict types/units/JSON, duplicate-key/size guard, eligibility/domain abstention, no-store, safe errors, readiness. Payload logging, persistence, demo fallback жоқ.
10. Fresh output workspace-та сол протокол толық қайталанып, evaluation JSON/таңдалған configuration/golden predictions дәл сәйкес шықты. Бұл жаңа орта не сыртқы клиникалық когорта емес.
11. 47 software tests өтті, failures/errors/skips 0. Дерек, guard, training/calibration/evaluation/export roundtrip, corrupt bundle, unavailable, wrong units/types, duplicate/large payload, unsupported cohort және invalid probability қаралды.
12. Нақты loopback HTTP арқылы 30 synthetic request: 30 success, 0 error, max reference difference 0, warm p95 2.46 ms. Уақытша server тексеруден соң тоқтатылды.

## Ғылыми шешім

2 027 адамның тек 49-ы positive. Final test positive count 10, calibration 5. Female test тобында positive 0: сол топтың discrimination сапасын бағалау мүмкін емес. Мұны «жақсы жұмыс істейді» деп түсіндіруге болмайды.

Таңдалған calibrated model AP=0.520548, AUROC=0.943939, Brier=0.016118. Glucose-only baseline AP=0.554714, AUROC=0.950884, Brier=0.015992. **Бес көрсеткішті панельдің қосымша пайдасы көрсетілген жоқ, MLP де жеңген жоқ.** Бұл ғылыми теріс нәтиже reproducible research artifact ретінде сақталады; clinical promotion жоқ.

Нақты көрсетілген шектеулер: retrospective single-examination outcome, complete-case fasting selection, medication skip uncertainty, female cohort coverage, аз positive count, бір internal split, клиникалық reference-standard және Қазақстан external validation жоқ.

## Бастапқы backlog мәртебесі

| Тармақ | Күйі | Дәлел / қалған тәуелділік |
|---|---|---|
| D1-01 task specification | Research target жазылды | [Model card](d1-model-card.md); клиникалық сарапшының approval-ы жоқ |
| D1-02 source audit | Орындалды | [Source manifest](../reports/d1/data_manifest.json), [cohort flow](../reports/d1/cohort_audit.json) |
| D1-03 protocol/split | Exploratory протокол орындалды | [Protocol](../reports/d1/protocol.json); clinical precision жеткіліктілігі расталмаған |
| D1-04 baselines | Орындалды | Dummy, glucose-only, regularized linear; [test report](../reports/d1/test_evaluation.json) |
| D1-05 HGB/MLP comparison | Орындалды | [CV selection](../reports/d1/selection.json); NN superiority жоқ |
| D1-06 calibration/threshold | Research calibration орындалды | [Calibration](../reports/d1/calibration.json); clinician-approved threshold жоқ |
| D1-07 final report | Орындалды, шектеулермен | [Model card](d1-model-card.md), [evaluation](../reports/d1/test_evaluation.json) |
| D1-08 bundle loader | Орындалды | [Manifest](../artifacts/d1/metadata.json), golden, corruption/parity tests |
| D1-09 Python API | Python бөлігі орындалды | [Contract](d1-api-contract.md); Java adapter пайдаланушы scope-ынан тыс |
| D1-10 frontend/PDF | Scope-тан шығарылды | Пайдаланушы frontend/backend-ті өзгертпеуді нақтылады |
| D1-11 integration/reproduction | ML HTTP және same-environment reproduction орындалды | [Runtime](../reports/d1/runtime_verification.json), [reproduction](../reports/d1/reproduction.json); browser/fresh-machine/басқа инженер review-ы жоқ |
| D1-12 external validation | Орындалған жоқ | Дербес рұқсат етілген reference-standard cohort және клиникалық/data partner қажет |

G0/G1 research scope анықталған, бірақ clinical approvals немесе clinical sample-size adequacy алынды деп мәлімделмейді. G2 бойынша hypothesized superiority расталмады. G3 Python бағдарламалық шекарасы тексерілді. G4 — сұралған public research code/evidence publication; clinical promotion емес. G5 орындалған жоқ.

## Тексеру дәлелі және қалған техникалық шектеулер

- [JUnit](../reports/d1/tests.xml): 47 test, 0 failure/error/skip. [Coverage](../reports/d1/coverage.json): бүкіл сақталған M2 + жаңа D1 кодының line coverage шамамен 53%; ол 80% деп көрсетілмейді. Нақты pipeline CLI бөлек орындалды; retained legacy M2 train/evaluate түгел қайта іске қосылған жоқ.
- Test warning-тері: legacy ONNX exporter deprecation, Starlette/AnyIO deprecated alias, sandbox-та physical core count анықталмауы. Бұлар test failure емес; жойылды деп мәлімделмейді.
- CI workflow қосылды; GitHub-та орындалған нәтижесі push-тан кейін ғана белгілі болады. Таза машина install және басқа platform-дегі real bundle compatibility әзірге тексерілмеді.
- Trusted joblib артефакты request-тен қабылданбайды. SHA matching provenance/signature-ді алмастырмайды; package build нұсқалары сәйкес келуі керек.
- Көп ауруға classifier, расталған диагноз, жаңа clinical data collection, LLM/chatbot, GPU/cloud, auto-retraining және аккаунт/тарих бұл орындалған зерттеудің бөлігі емес.

Келесі ғылыми қадам: clinical/data partner арқылы дұрыс reference-standard cohort және sample-size/precision жоспарын дайындау; жаңа дерек келгенге дейін осы viewed test-те tuning жасамау. Глюкоза-only baseline-нан артық ақпарат беретін белгі/label гипотезасы болса, ол жаңа preregistered experiment болады.
