# Task 3.5 - Pre-Model Audit Before Task 4

Audit date: 2026-09-26

## Scope and result

This audit read the existing Task 3 Parquet, schema, split, validation, leakage, correlation, and hard-case artifacts. It did not regenerate candidates or pair features and did not use test data.

| Audit area | Result | Evidence |
| --- | --- | --- |
| A. Feature subset | PASS | Exactly 66 recommended features; every feature exists in Parquet and has predictive role. |
| B. Leakage safety | PASS | No identifier, target, split, analysis-only, or ground-truth-derived field is recommended. |
| C. Entity split | PASS | 8,000 train and 2,000 validation S1 entities; zero overlap and zero pair-row split mismatches. |
| D. Numeric/data integrity | PASS | 2,499,947 rows scanned; zero null, NaN, or infinite recommended-feature values. |
| E. Hard-case inspection | PASS | 18 difficult positives and 18 hard negatives inspected; risks are modeling challenges, not dataset defects. |

**Task 3 artifacts are technically ready for Task 4 baseline modeling.**

## Recommended feature verification

- Recommended features: **66**.
- Stored dtypes: **23 bool, 30 float32, 9 uint16, 4 uint8**.
- Parquet/schema dtype mismatches: **0**.
- Missing recommended columns: **0**.
- Constant recommended features: **0**.
- Near-constant criterion: minority value prevalence below 0.5% for binary features. Only `address_exact` is flagged, true for **0.1352%** of pairs. This is sparse exact-match evidence, not leakage or a correctness defect.
- `eval_index`, `source1_entity_id`, `candidate_entity_id`, `match_group`, `split`, and `label` are all excluded.
- No other recommended field is derived from ground truth. `match_group` is used only to stratify the entity split and for analysis; it never enters model inputs.

The row-level data dictionary and feature-specific cautions are in `task3_to_task4_feature_audit.csv`.

## Inference and retrieval safety

The name, address, number, missingness, source, and interaction features are deterministic transformations of an S1 record and one retrieved candidate. Retrieval ranks and scores come only from the saved frozen Task 2.5 signal arrays. The label is computed afterward from ground truth and is not used in feature computation.

Retrieval features are inference-safe only if Task 4 and later inference use the unchanged frozen Task 2.5 retriever, signal definitions, baseline-preserving assembly order, and cap of 250. Per-signal rank `0` means absent from the saved top-250 signal and is paired with an explicit missing indicator. Per-signal score `0` uses the same missing convention. `frozen_candidate_position` is list position, not a calibrated rank.

## Redundancy review

The existing analysis reports seven high-correlation pairs where both features remain recommended:

| Feature A | Feature B | Spearman correlation |
| --- | --- | ---: |
| `name_token_set` | `transliterated_name_token_set` | 0.9770 |
| `number_address_rank` | `number_address_rank_missing` | -0.9757 |
| `number_address_rank_missing` | `number_address_score` | -0.9756 |
| `suffix_name_rank` | `suffix_name_rank_missing` | -0.9619 |
| `suffix_name_rank_missing` | `suffix_name_score` | -0.9619 |
| `transliterated_name_rank` | `transliterated_name_rank_missing` | -0.9551 |
| `transliterated_name_rank_missing` | `transliterated_name_score` | -0.9551 |

The retrieval rank/missing/score relationships are structurally expected and preserve explicit missing semantics. Baseline and transliterated token-set similarity are globally redundant but can diverge on cross-script records. These are review flags for Task 4 ablation, not reasons to alter Task 3.

`cross_script_name` is a deterministic non-ASCII mismatch proxy, not a true Unicode script classifier. It can also flag accented Latin names. The feature is inference-safe, but Task 4 interpretation should use that narrower definition.

## Train/validation split verification

- Train: **8,000 S1 entities**, **1,999,962 pairs**.
- Validation: **2,000 S1 entities**, **499,985 pairs**.
- S1 overlap: **0**.
- Pair rows with split inconsistent with their S1 mapping: **0**.
- Pair rows with match group inconsistent with their S1 mapping: **0**.
- Entity stratification: train has exactly **2,000** and validation exactly **500** entities in each of `1`, `2`, `3-5`, and `6+`.
- No S1 entity, and therefore no candidate pair for that S1 entity, crosses the split.

## Difficult-positive inspection

Metric abbreviations: `lev` is normalized Levenshtein, `set` is token-set similarity, `char3` is character 3-gram Jaccard, `J` is address-number Jaccard, and `signals` counts frozen retrieval signals in the top 100.

| # | S1 name | Candidate name | S1 address | Candidate address | Source | Name evidence | Address evidence | Numeric evidence | Retrieval | Why difficult |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | East Innovative Media LLP | ಈಸ್ಟ್ ಇನೋವೇಟಿವ್ ಮೀಡಿಯಾ ಎಲ್ಎಲ್‌ಪಿ | No-3350, First Floor, Banashankari 2Nd Stage, K.R Road, Bangalore, Karnataka | NO-1350, BENGALURU URBAN, Karnataka | S2 | lev 0.111, set 0.133, translit 0.441, suffix 0.074 | lev 0.366, set 0.533, char3 0.171 | both True, J 0.000, shared 0, conflict 4 | rank 30, signals 1, RRF 0.0111 | Task 2.5 recovery with cross-script name; Task 2.5-only recovery; cross-script; translit gain +0.33; number conflict; weak name |
| 2 | Rahul Ventures | Rahul Services | Ward No 09, Gulmohor Colony, Chambal Colony, Sheopur, Madhya Pradesh | (missing) | S2 | lev 0.643, set 0.714, translit 0.643, suffix 0.643 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 19, signals 1, RRF 0.0127 | Task 2.5 recovery with missing candidate address; Task 2.5-only recovery; candidate address missing; weak/missing address |
| 3 | Silver Constructions | सिल्वर कंस्ट्रक्शंस | C/O Sanbhaji Shivajirao Talap, Tal Walwa Dist. Sangli, Bhadkimbe, Sangli, Maharashtra | C/o Sanbhaji Shviajirao Talap, Tal Walwa Dist. Sangli, Bhadkimbe, Sangli, MH | S3 | lev 0.050, set 0.057, translit 0.214, suffix 0.050 | lev 0.863, set 0.920, char3 0.732 | both False, J 0.000, shared 0, conflict 0 | rank 8, signals 2, RRF 0.0294 | cross-script transliteration gain; cross-script; translit gain +0.16; weak name |
| 4 | LVD Touch Limited | LVD Tóuch Ltd | D-112, Magarpatta City, Pune City, Pune, Maharashtra | (missing) | S2 | lev 0.706, set 0.800, translit 0.765, suffix 0.889 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 14, signals 1, RRF 0.0135 | cross-script name; Task 2.5-only recovery; candidate address missing; cross-script; translit gain +0.06; weak/missing address |
| 5 | Petersen Banc | PETERSEN CENTER | 99 Belltown Road, Stamford, CT | (missing) | S2 | lev 0.667, set 0.762, translit 0.667, suffix 0.667 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 38, signals 1, RRF 0.0102 | missing address and weak address evidence; Task 2.5-only recovery; candidate address missing; weak/missing address |
| 6 | WO Energetics LLC | WO Energetics | 6809 Tarik Lane, Raleigh, NC | (missing) | S2 | lev 0.765, set 1.000, translit 0.382, suffix 1.000 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 4, signals 3, RRF 0.0469 | legal-suffix variation; candidate address missing; suffix gain +0.24; weak/missing address |
| 7 | Products Orion Sports Ltd | sportsorion.com | 102, Technopolis Knowledge, Park, Mahakali Caves Road, Mumbai, Maharashtra | 1-02, ANDHERI EAST, महाराष्ट्र | S2 | lev 0.400, set 0.400, translit 0.400, suffix 0.476 | lev 0.188, set 0.337, char3 0.011 | both True, J 0.000, shared 0, conflict 3 | rank 15, signals 3, RRF 0.0351 | conflicting address numbers; number conflict; weak name; weak/missing address |
| 8 | Davis, Shayne F., DDS, M.D., P.C. | Korlyra | 526 Meadowlake Lane, Lincoln, AL | Meadowlake Ln, Lincoln, Alabama | S3 | lev 0.038, set 0.061, translit 0.038, suffix 0.038 | lev 0.633, set 0.814, char3 0.559 | both False, J 0.000, shared 0, conflict 0 | rank 3, signals 2, RRF 0.0317 | weak name similarity; weak name |
| 9 | Delta Bright Silver | Delta Silver Center | 107 Third Street, Morgantown, WV | (missing) | S3 | lev 0.474, set 0.774, translit 0.474, suffix 0.474 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 26, signals 3, RRF 0.0278 | weak address similarity; candidate address missing; weak/missing address |
| 10 | Manzanares Motors | Manzanares Mffrgso | 1329 Broadway, Unit Unit 101, Fargo, ND | (missing) | S3 | lev 0.722, set 0.800, translit 0.417, suffix 0.722 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 6, signals 2, RRF 0.0303 | S3, 6+ match entity; candidate address missing; weak/missing address |
| 11 | Magma Retail Clinic | Magma Clinic (Center) | 8/201/A, Casa De Dios, Pazhavanchala, Pozhiyoor P O, Thiruvananthapuram, Neyyattinkara, N... | (missing) | S2 | lev 0.474, set 0.789, translit 0.474, suffix 0.474 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 10, signals 2, RRF 0.0286 | S2 single-match entity; candidate address missing; weak/missing address |
| 12 | Silver Constructions | सिल्वर कंस्ट्रक्शंस | C/O Sanbhaji Shivajirao Talap, Tal Walwa Dist. Sangli, Bhadkimbe, Sangli, Maharashtra | C/O SANBHAJI SHIVAJIRAO TALAP, TAL WALWA DIST. SANGLI, BHADKIMBE, SANGLI, महाराष्ट्र | S2 | lev 0.050, set 0.057, translit 0.214, suffix 0.050 | lev 0.863, set 0.931, char3 0.724 | both False, J 0.000, shared 0, conflict 0 | rank 4, signals 2, RRF 0.0306 | large transliteration improvement; cross-script; translit gain +0.16; weak name |
| 13 | Daniels Engineering Inc. | Daniels Inc. Center | 803 Circle Drive, Bethalto, IL | (missing) | S3 | lev 0.478, set 0.780, translit 0.348, suffix 0.526 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 25, signals 2, RRF 0.0214 | additional representative; candidate address missing; weak/missing address |
| 14 | Mulashi Technological Company | Mulashi Company 5ervice | Gate No-43, Dattawadi, Mulashi, Pune, Maharashtra | (missing) | S3 | lev 0.379, set 0.789, translit 0.379, suffix 0.435 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 25, signals 1, RRF 0.0118 | additional representative; candidate address missing; weak/missing address |
| 15 | Caban and Patten Commerce Inc | Caban and Patten - 6683788782 | Bagdad, KY, 210 Hyatts Store Road | (missing) | S2 | lev 0.586, set 0.744, translit 0.596, suffix 0.630 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 5, signals 2, RRF 0.0308 | additional representative; candidate address missing; weak/missing address |
| 16 | Kiser Legacy Entertainment | KISER LEGACY CENTER | 25020 Perdido Beach Boulevard, Unit Unit 103A, Orange Beach, AL | (missing) | S2 | lev 0.654, set 0.800, translit 0.654, suffix 0.654 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 15, signals 1, RRF 0.0133 | additional representative; Task 2.5-only recovery; candidate address missing; weak/missing address |
| 17 | Select Piedmont Nmp Co | Select Nmp Co Center | 23502 11th Terrace, Independence, MO | (missing) | S2 | lev 0.409, set 0.810, translit 0.409, suffix 0.450 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 46, signals 1, RRF 0.0094 | additional representative; Task 2.5-only recovery; candidate address missing; weak/missing address |
| 18 | Belcher Seas of Albany | BELCHER SEAS OF CENTER | 93 B Shaker Road, Unit Apartment 1A, Albany, NY | (missing) | S2 | lev 0.727, set 0.811, translit 0.727, suffix 0.727 | lev 0.000, set 0.000, char3 0.000 | both False, J 0.000, shared 0, conflict 0 | rank 9, signals 2, RRF 0.0288 | additional representative; Task 2.5-only recovery; candidate address missing; weak/missing address |

Recurring patterns are missing candidate addresses, cross-script/transliteration variation, business names that retain only a subset of tokens, legal-form changes, numeric conflicts, and Task 2.5-only recoveries with weak raw evidence. A future classifier must not require both name and address to be strong.

## Hard-negative inspection

| # | S1 name | Candidate name | S1 address | Candidate address | Source | Name evidence | Address evidence | Numeric evidence | Retrieval | Why difficult |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Binapani Educational Society | Binapani Educational Society Holdings | 2Nd Floor, Office No- 22 Arora Towers, Imoledina Road, Pune, Maharashtra | #ND FLOOR, OFFICE NO- 22 ARORA TOWERS, IMOLEDINA ROAD, PUNE, Maharashtra | S2 | lev 0.757, set 1.000, translit 0.757, suffix 0.757 | lev 0.985, set 0.992, char3 0.985 | both True, J 0.500, shared 1, conflict 1 | rank 1, signals 5, RRF 0.0807 | near-identical name and address; name threshold would fire; address threshold would fire; numbers overlap; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 2 | High's Distribution | High's Distribution | 670 Lake Avenue, Unit CONDO 207, Village Of Lake Delton, WI | Natick, 100 Oak Saint, Massachusetts | S3 | lev 1.000, set 1.000, translit 1.000, suffix 1.000 | lev 0.179, set 0.353, char3 0.000 | both True, J 0.000, shared 0, conflict 3 | rank 1, signals 3, RRF 0.0492 | exact normalized name; name threshold would fire; conflicts can be overlooked; suffix stripping increases agreement |
| 3 | Jarlay Environmental of Rensselaer | jarlaya environmental of rensselaer | 751 Matheson Avenue, Rensselaer, IN | 751 MATHESON AVENUE, RENSSELAER, IN | S2 | lev 0.971, set 0.986, translit 0.971, suffix 0.971 | lev 1.000, set 1.000, char3 1.000 | both True, J 1.000, shared 1, conflict 0 | rank 1, signals 5, RRF 0.0786 | exact normalized address; name threshold would fire; address threshold would fire; numbers overlap; multi-signal retrieval looks convincing |
| 4 | Ridgeline Atlantic | RIDGELINE ATLANTIC PARTNERS | 91 Hodge Avenue, Ansonia, CT | 98 HODGE AVE, ANSONIA, CT | S2 | lev 0.667, set 1.000, translit 0.667, suffix 0.667 | lev 0.846, set 0.898, char3 0.654 | both True, J 0.000, shared 0, conflict 2 | rank 1, signals 5, RRF 0.0807 | high name/address with numeric conflict; name threshold would fire; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 5 | Gee Direct Minerals LLC | Knotts Direct Minerals LLC | 2010 Mac Arthur Road, Unit Unit A, City Of Waukesha, WI | 2010- Mac Arthur Rd, # Unit A, City Of Waukesha, Wisconsin | S3 | lev 0.769, set 0.905, translit 0.784, suffix 0.727 | lev 0.731, set 0.909, char3 0.745 | both True, J 1.000, shared 1, conflict 0 | rank 2, signals 5, RRF 0.0635 | high name/address with shared numbers; name threshold would fire; address threshold would fire; numbers overlap; multi-signal retrieval looks convincing |
| 6 | Steed Excavation Inc | Steed-Excavation Co | 80 Cane Creek Lane, Dyersburg, TN | 81 Cane Creek Ln, Dyersburg, Tennessee | S3 | lev 0.850, set 0.923, translit 0.850, suffix 1.000 | lev 0.722, set 0.836, char3 0.525 | both True, J 0.000, shared 0, conflict 2 | rank 1, signals 5, RRF 0.0777 | legal-form normalization creates strong agreement; name threshold would fire; conflicts can be overlooked; suffix stripping increases agreement; multi-signal retrieval looks convincing |
| 7 | High's Distribution | High's [Distribution] | 670 Lake Avenue, Unit CONDO 207, Village Of Lake Delton, WI | 801 Tekolste Dr, Firth, Nebraska | S3 | lev 1.000, set 1.000, translit 1.000, suffix 1.000 | lev 0.161, set 0.296, char3 0.013 | both True, J 0.000, shared 0, conflict 3 | rank 2, signals 3, RRF 0.0484 | very high name but weak address; name threshold would fire; conflicts can be overlooked; suffix stripping increases agreement |
| 8 | Upper College | Eydie's Smart | WA, 13317 Shore, Nine Mile Falls | NINE MILE FALLS, 13317 SHORE, WA | S2 | lev 0.077, set 0.308, translit 0.160, suffix 0.077 | lev 0.100, set 1.000, char3 0.750 | both True, J 1.000, shared 1, conflict 0 | rank 1, signals 2, RRF 0.0328 | very high address but weak name; address threshold would fire; numbers overlap |
| 9 | Gladys Deaton Horizon Cms Inc | Gladys Deaton Horizon Cms West Inc | 904 Wythe Road, Springfield, IL | 11 WYTHE RD, SPRINGFIELD, IL | S2 | lev 0.853, set 1.000, translit 0.853, suffix 0.833 | lev 0.828, set 0.873, char3 0.645 | both True, J 0.000, shared 0, conflict 2 | rank 1, signals 5, RRF 0.0802 | retrieved by all five signals; name threshold would fire; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 10 | Sunrise Impex Private Limited | Sunrise Impex Limited | Flat No.B2, Athulayam Apartment, New No.2&4, Vasan Street, Chennai, Tamil Nadu | FLAT NO.B23, ATHULAYAM APARTMENT, NEW NO.2&4, VASAN STREET, CHENNAI, Tamil Nadu | S2 | lev 0.724, set 1.000, translit 0.724, suffix 1.000 | lev 0.986, set 0.993, char3 0.930 | both True, J 1.000, shared 2, conflict 0 | rank 1, signals 5, RRF 0.0658 | shared numbers without conflict; name threshold would fire; address threshold would fire; numbers overlap; suffix stripping increases agreement; multi-signal retrieval looks convincing |
| 11 | Diehl, Stewart and Pinkerton LLC | Diehl, Stewart and Pinkerton Corp Service #14183 | 5501 Campo Real Circle, Brownsville, TX | 05503 CAMPO REAL CIR, PMB 6199, BROWNSVILLE, TX | S2 | lev 0.630, set 0.931, translit 0.565, suffix 0.587 | lev 0.750, set 0.840, char3 0.551 | both True, J 0.000, shared 0, conflict 3 | rank 1, signals 5, RRF 0.0809 | multiple conflicting numbers; name threshold would fire; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 12 | Phillips, Hayes & Acampora Focus Center | Phillips, Hayes & Acampora Focus Center Central [Inc] | 4150 Lakeside Drive, Sellersburg, IN | 4159 Lakeside Dr, Sellersburg, Indiana | S3 | lev 0.750, set 1.000, translit 0.750, suffix 0.818 | lev 0.750, set 0.857, char3 0.571 | both True, J 0.000, shared 0, conflict 2 | rank 2, signals 5, RRF 0.0692 | S3 hard negative; name threshold would fire; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 13 | Historical Institute of Alexandria City | Historical Institute Of Alexandria City Corp | 5661 A Derby Court, Alexandria City, VA | 5662 A DERBY CT, ALEXANDRIA, VA | S2 | lev 0.886, set 1.000, translit 0.886, suffix 1.000 | lev 0.757, set 0.848, char3 0.512 | both True, J 0.000, shared 0, conflict 2 | rank 1, signals 5, RRF 0.0799 | S2 hard negative; name threshold would fire; conflicts can be overlooked; suffix stripping increases agreement; multi-signal retrieval looks convincing |
| 14 | Nelia's Atlantic Architecture | NELIA'S ATLANTIC ARCHITECTURE CORP | 390 1st Street, Rockaway Beach, MO | 393 1RD STREET, ROCKAWAY BEACH, MO | S2 | lev 0.853, set 1.000, translit 0.853, suffix 1.000 | lev 0.906, set 0.906, char3 0.622 | both True, J 0.333, shared 1, conflict 2 | rank 2, signals 5, RRF 0.0664 | additional representative; name threshold would fire; address threshold would fire; numbers overlap; conflicts can be overlooked; suffix stripping increases agreement; multi-signal retrieval looks convincing |
| 15 | Puma Management Limited | Puma Management Pvt Ltd | Mr 1, Ikeva Venture And Knowledge Advisory Services Pvt Ltd, Level 3, Nsl Centrum, Serene... | MR 6, IKEVA VENTURE AND KNOWLEDGE ADVISORY SERVICES PVT LTD, LEVEL 3, NSL CENTRUM, SERENE... | S2 | lev 0.739, set 0.826, translit 0.739, suffix 1.000 | lev 0.919, set 0.948, char3 0.802 | both True, J 0.333, shared 1, conflict 2 | rank 2, signals 5, RRF 0.0796 | additional representative; address threshold would fire; numbers overlap; conflicts can be overlooked; suffix stripping increases agreement; multi-signal retrieval looks convincing |
| 16 | International Energy Private Limited | International Energy Solutions Private Limited | Tangar Exports Llp No.3 (Old No.2A/4), Ranipet, Vellore, Tamil Nadu | Ranipet, Tangar Exports Llp No.#3 (Old No.2a/7), TN | S3 | lev 0.783, set 1.000, translit 0.783, suffix 0.667 | lev 0.468, set 0.938, char3 0.587 | both True, J 0.500, shared 2, conflict 2 | rank 2, signals 5, RRF 0.0534 | additional representative; name threshold would fire; address threshold would fire; numbers overlap; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 17 | Jessika Watkins, CPA East Hampton | Jessika Watkins, CPA East Hampton Coastal | 5 Colchester Avenue, East Hampton, CT | CT, EAST HAMPTON, 18 COLCHESTER AVE | S2 | lev 0.800, set 1.000, translit 0.800, suffix 0.800 | lev 0.143, set 0.912, char3 0.641 | both True, J 0.000, shared 0, conflict 2 | rank 2, signals 5, RRF 0.0799 | additional representative; name threshold would fire; address threshold would fire; conflicts can be overlooked; multi-signal retrieval looks convincing |
| 18 | Lynnet's Networks | Lynnet's Networks Group | 1601 Summerall Lane, Unit 206, Chesapeake City, VA | 1606 Summerall Lane, # 206, Chesapeake City, Virginia | S3 | lev 0.739, set 1.000, translit 0.739, suffix 0.739 | lev 0.750, set 0.884, char3 0.636 | both True, J 0.333, shared 1, conflict 2 | rank 2, signals 5, RRF 0.0774 | additional representative; name threshold would fire; numbers overlap; conflicts can be overlooked; multi-signal retrieval looks convincing |

A simple similarity threshold would fail because many negatives have near-identical names, addresses, shared numbers, legal-suffix-normalized names, and support from four or five retrieval signals. Numeric conflicts sometimes separate them, but other cases remain genuinely ambiguous from the available fields and require joint evidence plus careful validation.

## Feature-family sanity notes

- Retrieval ranks/scores: label-independent frozen Task 2.5 evidence. Keep paired missing indicators and avoid treating rank 0 as best rank.
- `frozen_candidate_position`: valid retrieval provenance, but not a calibrated similarity rank.
- `task2_baseline_member`: valid Task 2 retrieval-membership indicator; its exact complement is intentionally excluded.
- `candidate_source_s3`: inference-safe, but Task 4 metrics should be reported separately for S2 and S3.
- `cross_script_name`: inference-safe non-ASCII mismatch proxy; do not interpret it as a verified script-family comparison.
- Transliteration gain: signed deterministic delta. Negative values are valid and do not mean missing.
- Legal-suffix features: inference-safe under the frozen conservative normalizer; monitor false agreement after suffix removal.
- Address-number features: zeros can mean no overlap or unavailable numeric comparison. Use them with number-presence indicators.
- Interactions: deterministic and interpretable. The continuous component features remain available.

## Final decision

No leakage risk or data-integrity defect was found. No recommended feature requires removal before modeling. The review flags are sparse `address_exact`, the seven high-correlation pairs, the approximate semantics of `cross_script_name`, source-specific calibration, and the requirement to preserve the frozen retrieval pipeline exactly.

The selected-column Parquet scan completed in 3.3 seconds. No large dataset was created or rewritten. No classifier, threshold tuning, candidate regeneration, test-data access, prediction, calibration, or submission work was performed.
