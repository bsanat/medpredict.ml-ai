# MedPredict ML агентіне арналған skills: таңдау және орнату есебі

Күні: 26 қазан 2026. Scope: project-local skills, агент нұсқаулықтары және ML execution plan. Агенттің model/reasoning параметрлері өзгертілмеді. Global integrations, automatic learning және cloud jobs қосылған жоқ.

## Қосылған және қайта қолданылатын дағдылар

| Skill | Дереккөз / тексерілген нұсқа | MedPredict-тегі қызметі | Күйі |
|---|---|---|---|
| `scikit-learn` | [K-Dense scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scikit-learn), skill 1.5 | Tabular preprocessing, pipeline, CV, классификация және бағалау анықтамалары | `.agents/skills/scikit-learn/` ішінде орнатылды |
| `scientific-critical-thinking` | [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scientific-critical-thinking), skill 1.5 | Ғылыми claim, bias, статистикалық интерпретация және evidence appraisal | `.agents/skills/scientific-critical-thinking/` ішінде орнатылды |
| `medpredict-ml-engineering` | Осы тапсырмада жоба үшін жазылды | Task/label contract, M2 preservation, leakage, selection/test, bundle/API, evidence reporting | `.agents/skills/medpredict-ml-engineering/` ішінде жасалды |
| `ecc:mle-workflow` | Орнатылған ECC 2.2.3; [upstream workflow](https://github.com/affaan-m/ECC/blob/25278e7d7b6cf6e6a4e7a6c8e10f14ff22f0eb5b/skills/mle-workflow/SKILL.md) | Деректен serving/rollback-қа дейінгі lifecycle; жоспарды нақты deliverable-дарға бөлу | Бар plugin қайта пайдаланылады; көшірме орнатылмады |
| `ecc:pytorch-patterns` | Орнатылған ECC 2.2.3; [upstream skill](https://github.com/affaan-m/ECC/blob/25278e7d7b6cf6e6a4e7a6c8e10f14ff22f0eb5b/skills/pytorch-patterns/SKILL.md) | MLP, training loop, eval/no-grad, seed және shape тәртібі | Training кодын өзгерткенде ғана жүктеледі |
| `ecc:python-testing` | Қолданыстағы ECC skill catalog | Python data/metric/inference тесттері | Тест жазғанда қолданылатын бар skill; осы жұмыста іске қосылған жоқ |

Жоба үшін жазылған skill — жеке оқытылған AI модель не тәуелсіз background agent емес. Ол осы кодқа сәйкес жұмыс тәртібін бекітеді. Root `AGENTS.md` және `ml/AGENTS.md` тиісті ML тапсырмаларында оны оқуға бағыттайды. Келесі turn-да project-local skills discovery арқылы қолжетімді болуы тиіс; жаңа chat ашу міндетті емес. Discovery болмаған жағдайда root нұсқаулығындағы нақты жолмен оқылады.

## Қалай таңдалды

Алдымен skills.sh каталогы қаралды, кейін `npx skills find scikit-learn`, `scientific-critical-thinking`, `pytorch` арқылы ізделді. CLI-дегі snapshot: K-Dense scikit-learn шамамен 2.2K, critical-thinking 2.4K, ECC pytorch-patterns 10.2K installs. Бұл өзгермелі каталог сандары, сапаға немесе қауіпсіздікке кепіл емес. GitHub API бойынша сол тексерудегі K-Dense репозиторийі 54 749 star, repository license MIT; әр skill frontmatter license-ы бөлек сақталды.

SKILL.md мәтіндері, reference құрылымы және sklearn-дің екі example script-і қаралды. Тек іздеу рейтингіне сүйеніп ұсынылған жоқ. Skill кеңестерінің MedPredict контекстіне сай келмейтін жерлері root/ML нұсқаулықтарымен шектелді.

## Маңызды бейімдеулер

1. **Нұсқа:** sklearn skill 1.9.1-ге арналған; MedPredict 1.7.2-ге pinned. Оның install/upgrade командалары орындалмайды. API мысалдары [1.7 docs](https://scikit-learn.org/1.7/) бойынша тексеріледі. Example classifier-дегі `l1_ratio` параметрі мен өзге жаңа API-ларды жобаға сол күйінде көшіруге болмайды; бұл demo-лар compatibility-тестіленген MedPredict компоненттері емес.
2. **Host metadata:** екі upstream SKILL.md ішіндегі top-level `compatibility` мәтіні өзгеріссіз `metadata.compatibility` ішіне ауыстырылды, себебі осы Codex validator-ының accepted frontmatter тізімінде top-level өріс жоқ. Workflow body, references және example scripts өзгертілмеді.
3. **Лицензия/provenance:** upstream revision `5f2bc0341354fd5590aef76a533ada1e1e5b4e62`; бастапқы 20 файл Git blob hash арқылы source-қа сәйкес екені тексерілді. Екі шағын metadata бейімдеуі lock-та жазылады. Repository `LICENSE.md` әр vendor skill ішінде `UPSTREAM-LICENSE.md` болып сақталды; sklearn frontmatter BSD-3-Clause, critical-thinking MIT декларациялары сақталды.
4. **Сыртқы сервистер:** critical-thinking-тің optional OpenRouter schematic жолы бұл жобаға қажет емес және іске қосылмайды. Skill орнату дерек жіберуге рұқсат бермейді.
5. **Медициналық әдіснама:** prediction-model зерттеулерінде TRIPOD+AI және тиісті prediction-model bias assessment қолданылады. Intervention/RCT checklist-терін кез келген ML моделіне формалды түрде тықпаламаймыз.

Нақты source/local SHA-256 тізімі: [sources.lock.json](../../.agents/skills/sources.lock.json). Бұл lock сырттан жаңа кодты автоматты жүктемейді; update қажет болғанда жаңа revision жеке қаралады.

## Әзірге қосылмағандары

- [Hugging Face skills](https://github.com/huggingface/skills/tree/fabc634a893f62ad5c2039830397efa35fa62715): репозиторийдің бар екені және Apache-2.0 metadata тексерілді. Қазіргі MedPredict міндеті LLM fine-tuning/Hub publication емес; credential/cloud surface қазір қажет емес. Жеке skills мазмұнына толық audit жасалмады.
- K-Dense `pytorch-lightning`: каталогта табылды, бірақ қолданыстағы шағын PyTorch MLP үшін жаңа training framework қажет емес; орнатылмады.
- Бүкіл scientific skills bundle, GPU optimization, multi-agent orchestration, automatic memory/learning: нақты міндетсіз орнатылмады.

## Тексерулер және шегі

- Үш SKILL.md bundled `quick_validate.py` арқылы өтті. Validator-ға жетіспеген PyYAML 6.0.2 тек `tmp/ml-agent-skills-review/validator-deps` ішіне орнатылды; ML environment және оның lock-файлы өзгертілмеді.
- Upstream source bytes 20 файл бойынша тексерілді; metadata адаптациясынан кейін original/local hashes lock-та бөлек жазылады.
- Жергілікті Markdown сілтемелерінің бар болуы, YAML metadata, JSON lock және екі bundled Python example-дің синтаксисі тексерілді. Example scripts орындалмады; олар жобаға енген pipeline емес.
- Мәтін/скрипттерде сыртқы network/process/credential операцияларына қатысты үлгілер қаралды. Dedicated security scanner, dynamic sandbox analysis және барлық ғылыми reference-тің толық фактчекингі орындалған жоқ. Статикалық тексеру қауіпсіздік сертификаты емес.
- Агенттің нақты сапасын before/after benchmark арқылы өлшеу орындалмады. Алты behavior acceptance scenario негізгі жоспардың 17-бөлімінде берілген; олар өткен тесттер ретінде көрсетілмейді. Skills орнату модель accuracy-сін не клиникалық жарамдылықты арттырды деген тұжырым жасалмайды.

## Қолдану

Мысал: `$medpredict-ml-engineering D1 үшін дерек аудитін жасап, label leakage пен split протоколын тексер.`

Агент тапсырмаға сәйкес бір негізгі workflow және қажет анықтамаларды ғана оқиды. Орнату токен үнемдеуіне кепіл емес; қажетсіз skill loading азайтылған, бірақ сандық тиімділік өлшенбеген.

Негізгі жоспар: [диагностикалық ML execution plan](diagnostic-ml-plan-2026-10-26-kk.md).

Ғылыми software attribution: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). Авторлар мен metadata arXiv-тен тексерілді; сілтеме негізгі жоспарға да қосылды.
