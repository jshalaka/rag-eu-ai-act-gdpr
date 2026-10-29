# Source PDFs

The parser was exercised against these files in this repository:

| Filename | Document | Observed PDF pages |
|---|---|---|
| `full.pdf` | GDPR, Regulation (EU) 2016/679 | 117 |
| `TA-9-2024-0138_EN.pdf` | European Parliament AI Act first-reading text, 13 March 2024, P9_TA(2024)0138 | 459 |

The AI Act source is not the final Official Journal text of Regulation (EU) 2024/1689.
Article/recital numbering, wording and layout can differ. The current evaluation applies to
the supplied PDF corpus. Changing source versions requires fresh ingestion and review of
question expectations, chunk references and baseline provenance.

These two PDFs are tracked in this repository; exported copies may omit them. If a PDF is
missing, obtain the corresponding document from EUR-Lex or the European Parliament and verify
its document/version and layout before treating it as equivalent. The observed page counts
are descriptors, not cryptographic identity checks.

From the project root, `rag-eu-law ingest` reads PDFs and generates chunks and search indexes
under `data/processed/`. Generated files are not versioned. The corpus validator checks
readability and page bounds; manual comparisons are still needed for extraction fidelity.
See the [evaluation runbook](../../docs/evaluation/RUNBOOK.md).
