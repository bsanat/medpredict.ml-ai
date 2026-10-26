# Сравнение данных для MedPredict-SBP-v2

Проверено 21 октября 2026 года по официальным документациям. Решение относится к исследовательской оценке текущего систолического давления по возрасту, полу, ИМТ, общему холестерину и глюкозе плазмы натощак. Оно не означает клиническую пригодность модели.

## Кандидаты

| Критерий | NHANES 2017–March 2020 | NHANES August 2021–August 2023 | scikit-learn Diabetes |
|---|---|---|---|
| Официальный источник | [CDC: каталог pre-pandemic](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020) | [CDC: каталог 2021–2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023) | [Документация scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset), [исходное описание NCSU](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) |
| Объём исходной публикации | 15 560 записей demographics всех возрастов по [P_DEMO](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm); это не число взрослых с пятью измерениями | 11 933 записей demographics всех возрастов по [DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm); 8 860 записей обследования в [BMX_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm) | 442 пациента с диабетом, 10 исходных признаков |
| Нужные входы | Все пять доступны, но в разных файлах и у разных подвыборок | Соответствующие поля доступны в файлах нового цикла; нельзя считать всех участников complete case | Есть возраст, пол, ИМТ и лабораторные признаки, но также давление как predictor; набор входов и шкал отличается |
| Цель | Среднее нескольких осциллометрических SBP одного обследования, мм рт. ст.; не будущее событие | Аналогичные повторные SBP одного обследования; кандидат для отдельной проверки по более позднему периоду | Числовой показатель прогрессирования диабета через год; это не SBP и не вероятность диабета |
| Missingness | После проверенного чтения XPT: конечная когорта 3 318, train/validation/test = 2 324/497/497; полный учёт исключений приведён в [решении](dataset-selection.md) и [локальном аудите](../reports/data_audit.json). Отдельная частота пропуска в codebook не даёт размер пересечения | Наш join и аудит не выполнены; число подходящих полных взрослых наблюдений и доли пропусков не измерены | Missingness пакета в этом этапе не вычислялась; по малому размеру/наличию loader нельзя предполагать пригодность для нужной цели |
| Выборка | Сложная многоступенчатая выборка гражданского неинституционализированного населения США; объединённый pre-pandemic выпуск со специальными весами | Новый цикл с изменённым дизайном, вопросниками и процедурами после перерыва; отдельный аудит совместимости обязателен | Небольшая болезнь-специфичная исследовательская выборка; не источник популяционных весов NHANES |
| Воспроизводимость | Официальные XPT, codebooks, `SEQN`, веса и survey design; локальный manifest фиксирует SHA-256 | Официальные XPT/codebooks позволяют повторить будущую проверку; файлы этого цикла в текущий эксперимент не смешиваются | Встроенный loader удобен для обучения алгоритмам; стандартная выдача уже масштабирована, для своих экспериментов нужно учитывать `scaled` |
| Решение | Основные данные исправленного v2, при честном учёте отбора complete case | Резерв для будущей временной внешней проверки после фиксации модели | Не использовать для SBP-приложения; смена на него означала бы другую исследовательскую задачу и контракт |

NHANES 2019–March 2020 отдельно не является национально репрезентативным завершённым циклом. CDC объединил его с 2017–2018; дополнительное повторное включение 2017–2018 создало бы дубликаты источника. [Объяснение CDC](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020).

Первоначальный запуск с 3 470 участниками отозван: pandas 2.2.3 неверно прочитал нулевые XPT-значения как положительные `2**-260`, нарушив fasting eligibility. После независимого сравнения с pyreadstat 1.3.1 исходные данные читаются корректно. Архив сохранён локально и исключён из Git; публичное объяснение — [история отзыва v1](withdrawn-v1-status.md). Это техническое исправление не является поиском более удобного датасета по тестовой метрике.

## Компоненты и прямые источники

Для основного цикла выбраны следующие файлы; codebooks содержат определения полей и частоты, XPT используются pipeline напрямую:

| Компонент | 2017–March 2020 | 2021–2023, кандидат для будущей проверки |
|---|---|---|
| Возраст, пол, дизайн | [P_DEMO codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm), [XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt) | [DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm) |
| ИМТ | [P_BMX codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.htm), [XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.xpt) | [BMX_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm) |
| Общий холестерин | [P_TCHOL codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_TCHOL.htm), [XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_TCHOL.xpt) | [TCHOL_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.htm) |
| Глюкоза плазмы натощак | [P_GLU codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GLU.htm), [XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GLU.xpt) | [GLU_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm) |
| Повторные SBP/DBP | [P_BPXO codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.htm), [XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.xpt) | [BPXO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.htm) |
| Контекст лечения | [P_BPQ codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.htm), [XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.xpt) | Внешний цикл потребует отдельного согласования определений/skip patterns |

Наличие лабораторной записи ещё не означает соответствие fasting-подвыборке. В `P_GLU` 5 090 записей возраста 12+, из них 4 476 имеют положительный `WTSAFPRP`, а глюкоза измерена у 4 744. Разница включает участников вне fasting-критерия; эти числа нельзя объявлять итоговой взрослой обучающей когортой. [P_GLU](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GLU.htm).

В новом цикле fasting-вес называется `WTSAF2YR`. В `TCHOL_L` дополнительно появился `WTPH2YR` для phlebotomy nonresponse, а в течение цикла менялось лабораторное оборудование; CDC сообщает, что результаты холестерина корректировать не требовалось. При внешней проверке нужно следовать документации именно нового цикла, не механически переносить имена весов и поправки. [GLU_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm), [TCHOL_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.htm).

## Условия использования и ограничения

Для обоих NHANES применяется [NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html): использование для статистического анализа/отчётности, запрет установления личности и связывания с персонально идентифицируемыми сведениями. Публичный доступ не следует описывать как безусловную MIT/CC0-лицензию. В проекте сохраняются источники и агрегированные отчёты; исходные/обработанные строки исключены из Git.

Для Diabetes официальная страница источника предоставляет данные и библиографию, но отдельная лицензия именно массива данных в проверенном описании не установлена. Лицензия программного кода scikit-learn сама по себе не доказывает условия каждого включённого датасета. Этот массив не выбран и не используется для обучения SBP.

Главные риски NHANES для приложения: отбор участников с полным набором измерений, fasting-подвыборка, неучтённые препараты и сопутствующие состояния, различия измерительных условий и населения. Survey weights не устраняют автоматически item nonresponse или смещение complete-case отбора. Более поздний NHANES даст полезную временную проверку в США, но не проверку на населении Казахстана. Работа с cross-sectional записями не создаёт наблюдений изменения давления во времени.
