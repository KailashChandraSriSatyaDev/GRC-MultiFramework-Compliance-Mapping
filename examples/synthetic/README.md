# Synthetic demonstration data

These files are **invented** for a software-pipeline demonstration. They are not ISO/IEC 27001
text, not organisational policies and not research data.

- `demo_controls.csv`: 8 neutral controls, IDs `DEMO-01` to `DEMO-08`.
- `demo_statements.csv`: 12 neutral statements, IDs `DS-001` to `DS-012` (one, `DS-011`, is
  deliberately unrelated to security).

`python -m grc_mapper demo` maps them with `FakeEncoder`, a hash-seeded stand-in for the sentence
encoder. Its similarity scores carry no semantic meaning: the demo shows that loading, validation,
ranking, banding, DI/APS arithmetic, the priority queue, export and provenance work end to end.
The demo scores are not NLP inference and are not research results.
