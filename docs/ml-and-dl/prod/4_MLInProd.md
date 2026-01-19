# Planning and Scoping ML Projects (Part 2)

> Continuation of Lecture 3. Focus: Effective planning strategies, evaluating feasibility, and scoping research.

---

## Quick Recap (Lecture 3)

- **Why planning matters** → Misalignment = wasted effort
- **"Why are we building this?"** → The key question to ask stakeholders
- **Common misunderstandings**: Business knowledge, data quality, functionality, curse of knowledge, analysis paralysis

For details, see [3_MLInProd.md](./3_MLInProd.md)

---

## Planning Through Stakeholder Questions

### The Right Questions to Ask Business Team

| Question | What You Learn |
|----------|----------------|
| "How do you feature products on site today?" | Inventory rules, current manual process |
| "Do you have any rules you follow?" | Sponsored items, bestseller logic, deduplication |
| "What happens if we don't promote agreed items?" | Supplier contracts, revenue implications |
| "What do you mean by 'trending'?" | Social media signals, influencer data sources |

### Example: E-Commerce Recommendation Scoping

**From the discussion, ML team identified:**

| Feature | Priority | Notes |
|---------|----------|-------|
| Inventory checks + deduplication | **Critical** | Must not recommend out-of-stock or duplicates |
| Sponsored/contract items on top | **Critical** | Revenue agreements with suppliers |
| Global trends (not just weekly) | **Improvement** | Better than weekly-based recommendations |
| Social media trend analysis | **Defer** | Too complex for MVP |
| "Looks nice" aesthetic scoring | **Defer** | Subjective, needs research |

> Simple follow-up questions → Massive clarity on what actually matters.

---

## Why Demos Matter (Agile vs Waterfall)

### Without Demos (Waterfall Approach)
- Build all features → Show final product → Get feedback → Massive rework

### With Demos (Agile Approach)
- Build Feature 1 → Demo → Feedback → Build Feature 2 → Demo → Feedback → ...

### Case Study: Food Delivery Time Prediction

**Initial approach**: Calculate distance × speed = delivery time

**Stakeholder feedback during demo revealed missing features:**

| Feature | Status |
|---------|--------|
| Distance-based ETA | ✅ Implemented |
| Rain delay handling | ❌ Not considered |
| Traffic congestion | ❌ Not considered |
| Rider assignment delay | ❌ Not considered |

**Without demos**: All features done wrong → Full rework

**With demos**: Each feature validated → Caught issues early (e.g., "confidence interval for rain, not discrete time")

---

## Experimentation ≠ Solo Work

**Bad**: "I alone will solve this problem and present."

**Good**: Divide tasks → Communicate → Integrate → Present as team.

### The Bicycle Analogy

Three people design:
- Wheels
- Handle
- Seat

Without blueprint discussion → Components don't fit together.

**Fix:**
1. Meet business team early
2. Agree on approach
3. Test simple model outputs first
4. Show progress with early demos (demos might fail—that's fine)

---

## Scoping: The Fun Part

Scoping = Research phase where you validate feasibility.

### What Scoping Delivers

1. **Expected delivery date**
2. **Feasibility judgment**
3. **Risk assessment**

---

## Experimental Scoping: Time-Boxed Research

### The Principle

| Scenario | Approach |
|----------|----------|
| Unlimited time/money | Try every algorithm, build custom solutions |
| Real world | Time-box research, prioritize proven methods |

### Case Study: Churn Prediction Model

**Scope**: Telecom customer churn prediction

**Shortlisted algorithms**: Logistic Regression, XGBoost, Time-Based Cohort Analysis

**Time-boxed approach**:
- Week 1: Literature review for each algorithm
- Week 2: Demo implementation
- If not feasible → Discard and move on

> Time-boxing forces prioritization. You're not solving an infinite problem.

---

## Research Workflow

### Phase 1: Individual Exploration (1-2 days)
- Search blogs, white papers, company forums
- Gather broad solution ideas

### Phase 2: Deep Dive & Curation
- Each team member investigates viable methods in detail
- Compile refined list of technically valid options

### Phase 3: Group Discussion & Filtering
- Discuss findings as team
- Eliminate less promising options
- Narrow to 2-3 approaches for MVP

### Phase 4: Time-Boxed Prototyping
- Strictly time-limited implementation
- Focus on measurable outcomes, not polish

---

## Algorithm Selection Framework

### Classic vs State-of-the-Art

| Classic (Proven) | State-of-the-Art |
|------------------|------------------|
| SVD | Adversarial Networks |
| NMF | Deep RL |
| ALS | Hybrid Deep Learning |

**Decision**: For MVP, prefer classic + one promising SOTA approach.

### The Idea Board Method

Stick notes with algorithms → Add comments with:
- Summary of cool papers found
- **Risk level**: Low / Medium / High
- **Time to implement**: Low / Medium / High

| Risk | Time | Decision |
|------|------|----------|
| Low | Low | ✅ Try first |
| Low | Medium | ✅ Wonderful |
| Medium | Medium | ⚠️ Backup option |
| High | High | ❌ Discard |
| Low | "Who cares" (cool but impractical) | ❌ Discard |

---

## Prototyping: Not Production

**Purpose**: Compare feasibility and performance, not ship code.

### Prototype Evaluation Criteria

| Criteria | Why It Matters |
|----------|----------------|
| Speed to implement | Time is limited |
| Resource requirements | GPUs cost money |
| Performance gain | 10% vs 12% improvement = maybe not worth complexity |

### Avoid Resume-Driven Development (RDD)

**Bad**: "Let me use transformers/LLMs/agentic AI to show I know advanced stuff."

**Good**: "What's the simplest approach that solves the problem?"

> If Decision Tree gives 10% improvement and Transformer gives 12%, reconsider the effort.

---

## Visual Tools for Communication

### Why They Matter
- Technical diagrams bridge the gap between ML team and stakeholders
- Flowcharts show workflow without code

### Example: Library Management System Flowchart

```
Login → Verify → Valid?
           ├── Yes → [Browse | Return | Manage Readers | Manage Books | Query | Change Password]
           └── No → Error → Login
```

Stakeholders understand this. They don't understand gradient descent plots.

---

## Weighted Matrix: Unbiased Algorithm Selection

When team disagrees on approach, use weighted scoring.

### Steps

1. **List options** (SVD, Deep Learning, etc.)
2. **Define criteria** (Accuracy, Cost, Time, Maintainability)
3. **Assign weights** (e.g., Accuracy 40%, Cost 25%, Time 35%)
4. **Score each option** (1-5 scale)
5. **Calculate weighted score**
6. **Select highest scorer**

### Example

| Option | Accuracy (40%) | Cost (25%) | Time (35%) | **Total** |
|--------|----------------|------------|------------|-----------|
| SVD | 4 | 5 | 4 | **4.25** |
| Deep Learning | 5 | 2 | 2 | **3.20** |

**Winner**: SVD (higher weighted score despite lower accuracy)

---

## Time Is Everything

### Two Key Questions Business Cares About

1. **How long will this take?**
2. **How much will it cost?**

### For Unfamiliar/Risky Projects

- Limit research and experimentation time
- Communicate scope and risk clearly
- Prioritize simple solutions first

### By End of Experimentation Phase, Know:

- Overall scope
- Key features
- ETL needs
- Model inputs/outputs
- Supporting tools needed

---

## Quick Interview Points

1. **Planning through questions** = Ask "why" and "how" to stakeholders before building
2. **Demos** = Frequent validation beats big-bang delivery
3. **Time-boxing** = Research has deadlines, not infinite exploration
4. **Idea board** = Visual decision-making with risk/time assessment
5. **Weighted matrix** = Removes bias in algorithm selection
6. **RDD trap** = Don't pick tech to show off; pick what solves the problem
7. **Two questions** = "How long?" and "How much?"

---

## Recap

| Topic | Key Takeaway |
|-------|--------------|
| Stakeholder questions | Simple follow-ups reveal critical requirements |
| Agile demos | Validate features early, avoid massive rework |
| Time-boxed research | Treat scoping as guided sprint, not endless exploration |
| Algorithm selection | Idea board + weighted matrix = structured decisions |
| Prototyping | Speed and simplicity over sophistication |
| Communication | Flowcharts > gradient descent plots |

---

## Next Topics
- Building and validating MVP
- Translating prototypes into functional models
- Evaluation criteria and feedback loops
