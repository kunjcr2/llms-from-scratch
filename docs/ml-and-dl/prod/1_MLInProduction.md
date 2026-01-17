# Machine Learning in Production

> Kaggle accuracy ≠ Real-world performance. A 98% cat-vs-dog model can predict "mosquito net" with tiny noise added.

---

## Why Real-World ML is Different

- **Adversarial inputs**: Small perturbations can break models (stop sign → speed limit 100)
- **Scale**: 25K images on Kaggle vs billions in production
- **Stakeholder alignment**: Technical + communication skills matter equally
- **Beyond the model**: Scalability, deployment, long-term maintenance

---

## Why ML Projects Fail (Interview Gold)

| Cause | % | Key Point |
|-------|---|-----------|
| **Planning** | 30% | Vague ideas → vague outcomes |
| **Scoping/Research** | 25% | Wrong algorithm or insufficient research |
| **Technology** | 15% | Not enough compute/infra |
| **Fragility** | 15% | Edge cases not handled |
| **Cost** | 10% | Budget issues (less than you'd think!) |
| **Hubris** | 5% | Underestimating complexity |

> **Key insight**: Planning + Scoping = 55% of failures. Not cost, not tech.

---

## The "Just Enough" Rule

You don't need to master everything. Know enough about:
- **Software dev**: Clean, modular code + basic unit tests
- **Data engineering**: ETL pipelines, feature stores
- **Visualization**: Explain to non-technical stakeholders
- **Project management**: Timelines, scope, milestones

---

## Six Core Phases of ML Projects

### 1. Planning
- Talk to stakeholders FIRST
- Ask: "What's the real problem? How will predictions be used?"
- **Example**: Team thought NLP subject-line generator was needed. Client just wanted "best time to send emails"

**Key Questions**:
1. What's the real problem?
2. How is it done today?
3. How will my prediction be used?
4. What does success look like?

### 2. Scoping & Research
- **Too fast** (Team A): 1-day blog search → underestimate complexity → fail
- **Too slow** (Team B): Weeks on cutting-edge papers → no existing packages → over-budget
- **Balanced**: Time-bounded research + feasibility testing

### 3. Experimentation
- **Goldilocks problem**: Too few tests = miss best solution. Too many = waste time
- Rushed approach: Cherry-picked samples, adversarial cases break everything
- Over-ambitious: Analysis paralysis, exhausted compute budget
- **Middle ground**: Iterate in cycles, test incrementally

### 4. Development
- Write **testable, extensible, maintainable** code
- Use version control (Git/GitHub) for collaboration
- Avoid Jupyter notebook merge conflicts with proper branching

### 5. Deployment
- Batch vs Real-time predictions (cost implications)
- **Don't over-engineer**: Fancy ARIMA model might be too costly for simple forecasting
- Assess feasibility in planning phase, not after development

### 6. Evaluation
- Model performing well ≠ business impact
- Use **A/B testing** to prove value
- Monitor for drift, retrain as needed

---

## Waterfall vs Agile (Interview Topic)

### Waterfall Model
```
Requirements → Design → Development → Testing → Deployment
(rigid, no going back between phases)
```
- Linear, sequential
- Can't go back once a phase is complete
- Requirements locked early
- **Problem**: Late discovery of issues = expensive fixes

### Agile Model
```
Requirements → Design → Develop → Test → Review
    ▲                                 │
    │                                 │
    └─────────────────────────────────┘
         (iterate until ready)
```
- Iterative cycles (sprints)
- Continuous feedback and adaptation
- Requirements can evolve
- Frequent testing and stakeholder input
- **For ML**: Essential because data/models evolve unpredictably

> **Interview tip**: ML projects need Agile because models drift, data changes, and requirements clarify over time.

---

## Development Best Practices

### Bad: Single Notebook Chaos
- Two devs editing same notebook → merge conflicts
- No version control → lost work
- Can't test individual components

### Good: Git Workflow
1. Main branch = stable code
2. Feature branches for experiments
3. Pull requests for review
4. Merge only after tests pass

```
main ─────●─────────●─────────●──────
          │         ↑         ↑
feature-A └────●────┘         │
                              │
feature-B ─────────●──────────┘
```

---

## Batch vs Real-Time Deployment

| Aspect | Batch | Real-Time |
|--------|-------|-----------|
| Latency | Hours/Days | Milliseconds |
| Cost | Lower | Higher (APIs, infra) |
| Use case | Weekly reports | Live predictions |
| Complexity | Simple | REST APIs, scaling |

> Start with batch. Move to real-time only when business justifies the cost.

---

## A/B Testing (Evaluation)

Used to prove business impact:
1. Split users randomly into groups A and B
2. Show variant A to group A, variant B to group B
3. Measure KPIs (click rate, conversion, etc.)
4. Statistical analysis to determine winner

**Why it matters**: Even if model accuracy is high, stakeholders care about ROI.

---

## Quick Interview Points

1. **55% of ML failures** = poor planning + scoping
2. **Communication skills** as important as technical skills
3. **Agile over Waterfall** for ML projects
4. **"Just enough"** knowledge across domains beats deep expertise in one
5. **A/B testing** = standard for proving business value
6. **Batch first, real-time later** unless real-time is essential
7. **Version control** is non-negotiable for team collaboration

---

## Key Roles & Their Questions

| Role | Primary Concern |
|------|-----------------|
| Software Dev | Is the code clean? What algorithm? |
| Data Scientist | Why is the model drifting? |
| Model User | Can I trust predictions? |
| Business Stakeholder | What's the ROI? |
| Product Manager | What are model limitations? |
| Data Engineer | Is the data pipeline correct? |

---

## Next Topics (from lecture series)
- Data Scientist vs ML Engineer
- Risk reduction strategies
- DevOps vs MLOps
- Agile for ML projects
