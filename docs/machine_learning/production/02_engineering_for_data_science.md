# Your Data Science Could Use Some Engineering

> Data Scientist builds models. ML Engineer puts them in production. Both need each other.

---

## Data Scientist vs ML Engineer

### Data Science Focus
- **Feature Engineering**: Interpolation, encoding, vectorization, PCA
- **Algorithms**: CNN, RNN, NLP, supervised/unsupervised, clustering, time series, deep learning
- **Data Analysis**: Statistics, visualization, pattern extraction
- **Goal**: Extract patterns from data, develop optimized algorithms

### ML Engineering Focus
- **Project Planning**: Scoping, agile development, delivery commitments
- **Stakeholder Communication**: SME collaboration, requirement gathering
- **Monitoring & Evaluation**: Predictions, dashboards, A/B testing
- **Software Development**: OOP, functional programming, clean code
- **Model & Artifact Management**: Tracking, logging, repositories
- **Scalability**: Training pipelines, ML ETL, deployment
- **Production**: Cost, stability, CI/CD, integration testing

> **Key insight**: ML Engineering = Data Science + Deployment + Monitoring + Production workflows. DS builds the model, MLE makes it work at scale.

---

## ML is Technology + People + Process

| Component | What It Includes |
|-----------|------------------|
| **Technology** | Tools (sklearn, TensorFlow, PyTorch, HuggingFace), frameworks, algorithms |
| **People** | Business teams, IT, data engineers, domain experts |
| **Process** | Software dev standards, experimentation rigor, agile methodology |

---

## Simple First: The Complexity Slope

**Going down the slope increases**: Cost, complexity, maintenance difficulty, modification difficulty, explanation difficulty.

### Decision Flow (Interview Gold)

```
1. Is this actually a data science problem?
   NO → Communicate why it's not
   YES ↓

2. Can it be solved with visualization/aggregation/simple equations/UX change?
   YES → Solve it, stop
   NO ↓

3. Can heuristic/rule-based/descriptive analytics solve it?
   YES → Solve it, stop
   NO ↓

4. Can traditional ML (regression, classification, clustering) solve it?
   YES → Solve it, stop
   NO ↓

5. Can deep learning/graph-based models solve it?
   YES → Solve it, stop
   NO → Re-evaluate if it's a DS problem at all
```

### Case Study: Grocery Cart Abandonment

| Level | Solution |
|-------|----------|
| **Simple** | Bar chart of most-dropped items, simple UX button changes |
| **Rule-based** | If amount > ₹500 and no checkout in 5 min → send reminder |
| **Heuristic** | Time-of-day patterns, user-type patterns |
| **Traditional ML** | Logistic regression / Random Forest to predict abandonment likelihood |
| **Deep Learning** | Session-based clickstream models, co-purchase graph networks |

> **Interview tip**: Always start simple. Business doesn't care HOW you solved it—they care that it's feasible and cost-effective.

---

## Agile for ML Projects

### Why Agile Fits ML
- Iterative and incremental development
- Customer collaboration
- Continuous feedback loop
- Flexible to changing requirements

### Real-World Agile ML Examples

| Company | How They Use Agile |
|---------|-------------------|
| **Amazon** | A/B testing for recommendations, validate impact on KPIs |
| **Mayo Clinic** | Doctor-in-loop for patient risk models, iterate based on clinical feedback |
| **Uber** | Collaborate with ops team, test pricing models locally before full rollout |
| **Duolingo** | Simple interpretable models, A/B testing for streak features |
| **Walmart** | Incremental deployment of demand forecasting across locations |

---

## Two Key Agile Principles for ML

### 1. Communication & Cooperation

- ML work is complex—non-technical teams won't understand deep math
- Regular, clear discussions between ML team and business units
- Strong communication **within** the ML team itself
- **Working solo = school project. Real ML = teamwork.**

### Risky vs Agile-Aligned Approach

**Risky** (Never Works First Try):
```
Gather requirements → Build complete solution → Demo to customer → Hope it works
```

**Agile-Aligned**:
```
Gather requirements → Research options → Discuss with customer → Build feature → Show demo
    ↑                                                                      │
    └──────────────────── Iterate on failures ─────────────────────────────┘
```

> **Key difference**: Show demos, not final products. Prototype early, fail fast.

### 2. Embracing & Expecting Change

- Change is inevitable in ML projects
- Client may start wanting a basic calculator, later ask for scientific + currency exchange
- **Planning for change = designing modular, flexible systems**
- Without this mindset, even small changes require complete rebuilds

---

## DevOps vs MLOps

### What's Common
- CI/CD (build tools, continuous integration/deployment)
- Code development practices
- Software engineering standards
- Packaging
- Monitoring

### What MLOps Adds

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| **Agile** | Software agile | ML-aligned agile |
| **Monitoring** | Logging, failures | Logging + prediction accuracy + failure rates |
| **Release** | A/B testing | A/B testing + evaluation metrics |
| **Code Review** | Peer review, unit tests | + Metric validation (accuracy, F1, precision, recall) |
| **Environment** | Dev/QA/Prod | Dev/QA/Prod + simulation tests |
| **New Component** | — | Artifact management (registries, model tracking) |

> **Simple way to think about it**: MLOps = DevOps + everything related to model data & evaluation.

```
DevOps: Code → Build → Test → Deploy → Monitor
MLOps: Code → Build → Test → Deploy → Monitor
          ↑                           ↑
      + Model                    + Model Performance
      + Data                     + Data Drift
```

---

## Quick Interview Points

1. **Data Scientist** = extracting patterns from data
2. **ML Engineer** = deployment, monitoring, production workflows
3. **Start simple** → only increase complexity if simple methods fail
4. **Not every problem assigned to you is a DS problem** → validate first
5. **Agile for ML** = iterative feedback, collaboration, expect change
6. **MLOps ≠ DevOps** → MLOps adds model/data-specific concerns
7. **Communication skills** = as important as technical skills in ML projects

---

## Recap

| Topic | Key Takeaway |
|-------|--------------|
| DS vs MLE | DS extracts patterns, MLE deploys and maintains |
| Simple Approaches | Down the slope = more cost and complexity |
| Agile for ML | Iterative, collaborative, change-ready |
| DevOps vs MLOps | MLOps = DevOps + model & data lifecycle |

---

## Next Topics
- Planning and scoping in detail
- More technical deep-dives as series progresses
