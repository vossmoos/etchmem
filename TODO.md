# TODO — Future versions

> **Etchmem** — *The enterprise memory operating system for AI employees.*

This document captures ideas for future versions. No implementation commitments — just a place so we don't forget.

---

## Category

**AI Employee Memory OS**

This is the category we think is still available.

---

## Next versions (feature backlog)

- Quality scoring
- Bias checks
- Validation
- Versioning

### Memory capabilities

- Ownership
- Permissions
- Audit history
- Aging
- Conflicting memories
- Policies
- Inter-agent sharing

---

## The questions nobody has solved well

### 1. Memory governance

Example memory:

| Field | Value |
|-------|-------|
| Memory | "Customer ACME hates discounts" |
| Source | Sales call #123 |
| Confidence | 87% |
| Created | March 5 |
| Expires | after 1 year |
| Visible | sales agents only |
| Approved | human manager |

### 2. Organizational memory

```
Sales agent learns something
          |
          v
Company memory
          |
          v
Support agent benefits
          |
          v
Marketing agent adjusts campaigns
```

### 3. Memory lifecycle

Human memory does not work like a database. You need:

- Forgetting
- Reinforcement
- Contradiction resolution
- Confidence decay
- Promotion from temporary → permanent knowledge

### 4. Auditable memory

For enterprises: **Why did the AI make this decision?**

Because:

- Memory #1274
- Created by agent X
- Based on email Y
- Approved by manager Z

---

## Git layer for AI memory

Build the **Git layer for AI memory** — version control for company knowledge.

```
git commit memory
git blame memory
git diff memory
git revert memory
git branch memory
```

Features:

- Memory versions
- Approvals
- Merging conflicting memories
- Provenance
- Rollback
- Testing memories

---

## Connectors — knowledge ingestion

Develop connectors for data sources. Connect to:

- Gmail
- Outlook
- Slack
- Teams
- Confluence
- Notion
- Jira
- Salesforce
- HubSpot
- Google Drive
- SharePoint
- Databases

---

## Etch structure (entity model)

Create entities:

**Customer: ACME**

Properties:

- Annual revenue: $5M
- Uses product X
- Has migration concerns

History:

- 2024: failed migration
- 2025: new CTO interested in retry

People:

- CTO: Sarah
- Account owner: Mike

### Raw RAG vs etchmem

**Input** — Sales call transcript:

> "We don't want yearly contracts because our budget is uncertain."

**Raw RAG:** Store transcript.

**etchmem:**

| Field | Value |
|-------|-------|
| Customer | ACME |
| Fact | Rejects yearly contracts |
| Reason | Budget uncertainty |
| Confidence | 0.92 |
| Source | Sales call 2026-01-12 |

---

## Hard problems

### Contradiction handling

This is a huge unsolved problem.

Two memories:

| When | Memory | Confidence |
|------|--------|------------|
| 2025 | Customer wants monthly billing | 0.8 |
| 2026 | Customer signed annual contract | 0.95 |

**System decides:** Old preference replaced.

### Access control

Not every AI sees everything.

| Agent | Access |
|-------|--------|
| Sales agent | ✓ CRM, ✓ Sales emails |
| Support agent | ✓ Tickets |
| HR agent | ✓ Employee data |
| Finance agent | ✓ Invoices |

### Memory provenance

Every memory has:

- Who created it?
- Which document?
- Which agent?
- Which human?
- When?

### Knowledge versioning

**Git for company knowledge.**

**Knowledge v1.2**

Changed:

- Added new pricing rules
- Removed obsolete API documentation

Rollback available.

Imagine:

```
git diff company-knowledge
```

### Memory quality analytics

Dashboard: **Company Knowledge Health**

| Metric | Value |
|--------|-------|
| Total memories | 4.3 million |
| High confidence | 91% |
| Contradictions | 15,213 |
| Stale | 340,000 |
| Unknown areas | Customer onboarding |
