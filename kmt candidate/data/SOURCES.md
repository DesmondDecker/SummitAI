# Data sources and provenance

All pairs were machine-extracted from books you supplied (OCR / text layers) and are UNVERIFIED. Local-language text is ASCII-normalised (tone/length/dot marks and special letters removed) because OCR cannot read them reliably.

| source | what | language | note on rights |
|---|---|---|---|
| migeod1908 | F.W.H. Migeod, *The Mende Language* (1908): phrases, grammar examples, vocabularies, stories | Mende | 1908 text, generally public domain; check your jurisdiction |
| peacecorps_mende / _krio / _temne | Peace Corps language manuals (OCR) | Mende, Krio, Temne | US-government-produced; generally public domain, verify |
| visitsierraleone2006 | S. U. Kamarah, *Basic Krio and Temne Vocabulary*, visitsierraleone.org (2006), school-recommended orthography; hand-curated | Krio, Temne | (c) 2006 visitsierraleone.org; confirm permission before redistributing |
| mbsse_syllabus2020 | one example sentence from the Mende syllabus, Ministry of Basic and Senior Secondary Education (2020) | Mende | government curriculum document |
| jw_krio | ~35k English-Krio sentence pairs from Jehovah's Witnesses publications (training_data_krio3.txt), cleaned and alignment-filtered. **Religious domain only; rights unconfirmed (publisher copyright).** | Krio | excluded by `build_corpus.py --release` |
| vandevate2006 | M. van de Vate, *Prepositions in Krio* (interlinear examples) | Krio | copyrighted paper; short linguistic examples; check before redistributing |

Considered and NOT used for training: Stanford Alpaca (CC BY-NC 4.0 research-only data, generated with an OpenAI model, so not suitable for a commercial product), the Open-Assistant GitHub repo (code only; its chat data is hosted separately and is English), programming/English textbooks and project documents (English knowledge: better served by a retrieval knowledge base than by training this model).

Not used: the Berry Krio dictionary, `peace_corps_introductory_krio` and the Dwyer handbook (not yet OCR'd/extracted; see README), duplicate scans of Migeod and of the Temne manual.

## Counts

| language | source | split | pairs |
|---|---|---|---|
| krio | peacecorps_krio | heldout | 21 |
| krio | peacecorps_krio | train | 171 |
| mende | mbsse_syllabus2020 | train | 1 |
| mende | migeod1908:english_mende | train | 410 |
| mende | migeod1908:grammar_examples | train | 805 |
| mende | migeod1908:mende_english | train | 392 |
| mende | migeod1908:phrases | train | 718 |
| mende | migeod1908:stories | heldout | 204 |
| mende | migeod1908:stories | train | 1009 |
| mende | migeod1908:topical_vocab | train | 477 |
| mende | peacecorps_mende | heldout | 6 |
| mende | peacecorps_mende | train | 67 |
| temne | peacecorps_temne | heldout | 22 |
| temne | peacecorps_temne | train | 90 |

**RELEASE build.** The sources jw_krio, visitsierraleone2006 and vandevate2006 described above are NOT included in this build (copyrighted or rights unconfirmed). The remaining sources are believed to be public-domain or government-produced (Migeod 1908, Peace Corps manuals, a ministry syllabus) but you must verify that for your jurisdiction before publishing.
