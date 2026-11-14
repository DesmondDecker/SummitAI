# Krio specialist checkpoint

* 0.81 M parameters, d128 x 3 layers, 4,000 steps, best held-out checkpoint; tokenizer keeps the special letters (no ASCII fold).
* Trained on ~33.6k English-Krio pairs from Jehovah's Witnesses publications (`jw_krio`) plus ~330 pairs from the other Krio sources.
* **The `jw_krio` data has unconfirmed rights (publisher copyright). Do not publish these weights, or any corpus containing that data, without
  permission.** For a releasable build use `python tools/build_corpus.py --release`, which leaves that source out, and retrain. (Not legal advice.)
* Quality (held-out, spelling-insensitive chrF): in-domain 0.21 (Krio->English), 0.29 (English->Krio); out-of-domain 0.03-0.19. It does not translate
  reliably, so `calibration.json` makes the assistant refuse to translate unseen text. Language ID on in-domain Krio/English sentences is 100%.
* Use:  python chat.py --ckpt checkpoints_krio
