---
title: AI Agent Skills
---

## AI Agent Skills

Liken makes available agent skills for use in agentic workflows. This is an optional inclusion to your project, and will help you navigate the various APIs so as to best help you solve your problem.

Install the bundle from the [tessl](https://tessl.io/registry/victoraut/liken-skills) registry:

```bash
tessl install victoraut/liken-skills
```

The bundle contains one skill per API tier:

| Skill | Teaches |
| ----- | ----- |
| `liken` | Overview, and which API to reach for |
| `liken-dedupers` | Applying built-in dedupers |
| `liken-pipelines` | Pipelines with AND/OR/NOT rules and built-in preprocessors |
| `liken-custom-dedupers` | Writing your own dedupers in pure Python |
| `liken-record-linkage` | Canonicalization and synthetic records |
| `liken-backends-performance` | Backend selection, scaling and performance |

??? info "Using the skills"
    Once installed, agent-skill-aware tools (Claude Code, Cursor, and others) discover the skills automatically and load the relevant one on demand. Pin a version for reproducibility, e.g. `tessl install victoraut/liken-skills@0.1.0`. See the [tessl documentation](https://docs.tessl.io) for managing installed skills.
