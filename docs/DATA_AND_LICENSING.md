# Data and licensing

## Not included in this repository

| Asset | Reason |
|---|---|
| Fine-tuned model files (weights, tokenizer, configuration, model card) | not redistributed; the model card also embeds training examples |
| ISO/IEC 27001:2022 control descriptions (`Control_Text`) | copyrighted standard text; rights not cleared |
| The 206-policy research dataset | not redistributed (part of it has an unverified licence) |
| External evaluation corpora, annotation sheets, LLM label files | third-party or restricted text |
| Research notebooks and their outputs | outputs embed policy and control text |
| Intermediate and historical predictions | derived from restricted text |

The repository holds only code, tests, documentation, the approved environment lock, a runtime
model manifest (file names, sizes and SHA-256 hashes; no model files) and invented synthetic data.

## Synthetic data

`examples/synthetic/` contains 8 controls (`DEMO-01` to `DEMO-08`) and 12 statements (`DS-001` to
`DS-012`) written for the demonstration. They are not derived from ISO text or from the research
policies. The publication scanner checks public files for 8-word overlaps with the local control and
policy texts.

## Licence

Pending confirmation of university and co-author permissions. All rights reserved until a licence
file is added.
