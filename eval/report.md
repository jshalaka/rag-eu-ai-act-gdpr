# Evaluation report

| metric | value |
|---|---|
| questions (answerable / out-of-scope) | 20 / 6 |
| retrieval hit rate@5 | 100% |
| mean retrieval recall@5 | 84% |
| citation accuracy | 90% |
| refusal accuracy (out-of-scope) | 100% |
| false refusal rate (answerable) | 5% |
| mean latency | 5,873 ms |
| hit rate@5, direct_lookup (7 q) | 100% |
| hit rate@5, concept (7 q) | 100% |
| hit rate@5, cross_regulation (6 q) | 100% |

## Questions

| id | type | retrieval | citation | refused | question |
|---|---|---|---|---|---|
| d01 | direct_lookup | yes | yes | no | What does GDPR Article 17 say? |
| d02 | direct_lookup | yes | yes | no | What does Article 5 of the AI Act prohibit? |
| d03 | direct_lookup | yes | yes | no | What does GDPR Article 22 say about automated individual decision-making? |
| d04 | direct_lookup | yes | yes | no | How does Article 3 of the AI Act define an AI system? |
| d05 | direct_lookup | yes | yes | no | What does GDPR Article 33 require after a personal data breach? |
| d06 | direct_lookup | yes | yes | no | What transparency obligations does Article 50 of the AI Act impose? |
| d07 | direct_lookup | yes | yes | no | What does Recital 47 of the GDPR say about legitimate interests? |
| c01 | concept | yes | yes | no | When is a data protection impact assessment required? |
| c02 | concept | yes | yes | no | What are the lawful bases for processing personal data? |
| c03 | concept | yes | yes | no | What are the conditions for valid consent under the GDPR? |
| c04 | concept | yes | yes | no | What obligations do providers of high-risk AI systems have? |
| c05 | concept | yes | yes | no | Which AI systems are classified as high-risk? |
| c06 | concept | yes | yes | no | What fines apply when the AI Act's prohibited practices are infringed? |
| c07 | concept | yes | yes | no | What information must a controller give a data subject who asks for access to their personal data? |
| x01 | cross_regulation | yes | no | no | How does the AI Act refer to GDPR on biometric data? |
| x02 | cross_regulation | yes | no | yes | May providers of high-risk AI systems process special categories of personal data to detect bias, and how does this relate to GDPR Article 9? |
| x03 | cross_regulation | yes | yes | no | How does the fundamental rights impact assessment under the AI Act relate to a GDPR data protection impact assessment? |
| x04 | cross_regulation | yes | yes | no | Does the AI Act change how Union data protection law applies to personal data processed by AI systems? |
| x05 | cross_regulation | yes | yes | no | How does the AI Act's ban on real-time remote biometric identification relate to the GDPR rules on special categories of data? |
| x06 | cross_regulation | yes | yes | no | Which GDPR provisions are relevant to automated decisions made by high-risk AI systems about people? |
| o01 | out_of_scope | - | - | yes | What is the capital of France? |
| o02 | out_of_scope | - | - | yes | How do I bake sourdough bread at home? |
| o03 | out_of_scope | - | - | yes | What does the California Consumer Privacy Act say about the right to delete? |
| o04 | out_of_scope | - | - | yes | What are the penalties under the UK Data Protection Act 2018? |
| o05 | out_of_scope | - | - | yes | Who won the 2022 FIFA World Cup? |
| o06 | out_of_scope | - | - | yes | Does the Digital Services Act require platforms to remove illegal content within 24 hours? |

## Failures

- **x01** How does the AI Act refer to GDPR on biometric data?
  - retrieved: ['AI_ACT:art3:para36:seq0', 'AI_ACT:art5:para1:seq1', 'AI_ACT:art26:para10:seq1', 'AI_ACT:rec38:seq0', 'AI_ACT:art5:para7:seq0']
  - answer: "The AI Act refers to the GDPR on biometric data, particularly in the context of real-time remote biometric identification systems used for law enforcement in publicly accessible spaces. According to Recital 38 of the AI Act, the rules of this Regulation that prohibit, subject to certain exceptions, such use, which are based on Article 16 TFEU, should apply as lex specialis in respect of the rules on the processing of biometric data contained in Article 10 of Directive (EU) 2016/680 [AI Act Recital 38, p. 37]. This means that the AI Act's provisions on biometric data, especially in the context of law enforcement, are intended to be interpreted in conjunction with the GDPR, as the latter provides the legal basis for the processing of personal data. However, the AI Act does not provide the legal basis for the processing of personal data under Article 8 of Directive (EU) 2016/680, but rather complements the GDPR in regulating the use of real-time remote biometric identification systems for law enforcement purposes [AI Act Recital 38, p. 37]."; citations: ['[AI Act Recital 38, p. 37]']
- **x02** May providers of high-risk AI systems process special categories of personal data to detect bias, and how does this relate to GDPR Article 9?
  - retrieved: ['GDPR:art9:para4:seq0', 'GDPR:art9:para1:seq0', 'GDPR:art9:para3:seq0', 'GDPR:art9:para2:seq0', 'GDPR:art9:para2:seq1']
  - answer: 'Not covered by the provided documents'; citations: []
