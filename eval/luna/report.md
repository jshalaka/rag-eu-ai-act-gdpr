# Evaluation report

| metric | value |
|---|---|
| questions (answerable / out-of-scope) | 20 / 6 |
| retrieval hit rate@5 | 100% |
| mean retrieval recall@5 | 84% |
| citation accuracy | 85% |
| refusal accuracy (out-of-scope) | 100% |
| false refusal rate (answerable) | 5% |
| mean latency | 5,425 ms |
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
| x05 | cross_regulation | yes | no | no | How does the AI Act's ban on real-time remote biometric identification relate to the GDPR rules on special categories of data? |
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
  - answer: 'The provided documents do not explain how the AI Act refers to the GDPR on biometric data. They refer instead to Article 10 of Directive (EU) 2016/680 in a recital discussing real-time remote biometric identification for law enforcement; that recital is explanatory, not a binding article. [AI Act Recital 38, p. 37]'; citations: ['[AI Act Recital 38, p. 37]']
- **x02** May providers of high-risk AI systems process special categories of personal data to detect bias, and how does this relate to GDPR Article 9?
  - retrieved: ['GDPR:art9:para4:seq0', 'GDPR:art9:para1:seq0', 'GDPR:art9:para3:seq0', 'GDPR:art9:para2:seq0', 'GDPR:art9:para2:seq1']
  - answer: 'Not covered by the provided documents'; citations: []
- **x05** How does the AI Act's ban on real-time remote biometric identification relate to the GDPR rules on special categories of data?
  - retrieved: ['AI_ACT:rec17:seq0', 'AI_ACT:rec38:seq0', 'AI_ACT:art5:para7:seq0', 'AI_ACT:rec39:seq0', 'AI_ACT:art5:para4:seq0']
  - answer: 'The AI Act’s rules for real-time remote biometric identification in publicly accessible spaces for law-enforcement purposes operate as a specific framework for that use and the biometric-data processing it involves. [AI Act Recital 38, p. 37] Recital 38 explains that this framework applies as *lex specialis* in relation to the biometric-data rules in Article 10 of the Law Enforcement Directive; it does not state that the GDPR’s special-category rules govern that law-enforcement use. [AI Act Recital 38, p. 37]\n\nFor biometric identification outside that law-enforcement context, the AI Act recital says processing should continue to comply with the otherwise applicable requirements, and notes that GDPR Article 9(1) prohibits biometric-data processing subject to limited exceptions. [AI Act Recital 39, p. 38] This is an explanation in a recital, not a binding article. [AI Act Recital 39, p. 38]'; citations: ['[AI Act Recital 38, p. 37]', '[AI Act Recital 39, p. 38]']
