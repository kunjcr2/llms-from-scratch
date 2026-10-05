# Planning and Scoping ML Projects (Part 1)

> What really fails ML projects? Not algorithms, not data, not tools—it's poor planning and scoping.

---

## Why Planning Matters

If your start isn't proper, your end won't be either. Even with clear problem understanding, without proper research and literature survey, scoping will be inadequate and projects will fail.

**The cost of poor planning:**
- Waste of resources
- Waste of time
- Waste of money

### Case Study: Amazon's AI Recruiting Tool

Amazon built a tool to select top 5 resumes from 100. The model became biased toward male candidates because historically more male resumes existed in the database.

| What Went Wrong | Root Cause |
|-----------------|------------|
| Gender bias in selection | Data imbalance not addressed during scoping |
| Tool discarded | Lack of foresight about demographic distribution |

> Algorithm wasn't wrong. Data wasn't inherently wrong. **Scoping failed to anticipate** the data imbalance problem.

---

## The Paradigm Shift in ML

| Old Thinking | New Thinking |
|--------------|--------------|
| "How do we build it?" | "What are we solving and for whom?" |

### Case Study: IBM Watson for Cancer Care

IBM pitched Watson as a revolution in cancer care. Result? Hospitals rejected it because:
- Gave irrelevant suggestions
- Went against clinical safety measures
- Focused on generating text rather than patient safety

**Lesson**: In medical (and any user-facing) domain, the "for whom?" question is critical. Safety and user needs come first.

---

## Cross-Functional Planning = Better Outcomes

### Example: Spotify Wrapped

Spotify Wrapped succeeds because it involves:
- **Business managers**: Understand music industry trends
- **ML/Data Science team**: Build personalization algorithms
- **UI designers**: Create engaging, shareable visuals

Result: Users share their Wrapped on social media → free marketing + increased engagement.

### Benefits of Cross-Functional Planning
1. **Reduces experimentation time** - Can't waste months; business needs real solutions
2. **More business-ready solutions** - 100% accurate model ≠ business alignment
3. **Planned demos and deliverables** - Structured approach to delivery

---

## Planning Scenarios: Inadequate vs Adequate

### Scenario: Bank Fraud Detection

**Problem**: System creates false alarms for correct transactions, blocking legitimate purchases.

#### Inadequate Planning Approach

```
Understand high-level problem → Try Algorithm A → Add Feature X → Try Algorithm B
→ Add Feature X → Results bad → Try Algorithm C → Add Feature XYZ → Try Algorithm D
→ Add N symbol + merge logic → Try Deep Learning → Add Feature X+Y → "Choose least bad"
```

**Result**: Months wasted, compromise on "least bad" solution.

#### Adequate Planning Approach

```
1. Ask business team:
   - What was the earlier algorithm?
   - Why did it fail?
   - What's the time requirement? (e.g., 1 second response)

2. Kickoff meeting for scoping

3. Internal DS team meeting for research planning

4. Literature survey → Explore approaches

5. Test approaches in experimentation phase (2 weeks each):
   - Approach 1: Decision Tree
   - Approach 2: Rule-based
   - Approach 3: Deep Learning with embeddings

6. Meeting with business unit → Explain testing, gather feedback

7. Finalize approach based on discussion

8. Build features iteratively until MVP is complete

9. If not good → Rework based on feedback → Reiterate
```

**Key**: Integrate feedback loops + iterate on decisions. Business may prefer rule-based over deep learning if it's more convenient and correct.

---

## The "You Want Me to Predict What?" Problem

### Case Study: E-Commerce Recommendation System

**Business request**: Personalized recommendations for each user.

**What ML team did**: Built recommendation system, evaluated using RMSE plots and NDCG values.

**What business stakeholders wanted**: A live demo—"If I search for AC, will it recommend similar ACs?"

#### The Demo Disaster

Marketing analyst tested the system and found:

| Recommendation | Issue |
|----------------|-------|
| Product 1: Sandals | Okay |
| Product 2: Same sandals, different color | Duplicate—different product ID for same item |
| Product 3: Shoes (ID 234) | Product retired (ID < 1000 = out of stock) |
| Product 4: Same shoe, new ID | No handling for replacements |
| Products 5-7: All shoes, different colors | Only shoes? We're an e-commerce company, not shoe store |

**Root causes**:
1. Product catalog had duplicate IDs for color variants
2. No filter for retired products (ID < 1000)
3. Data science team assumed data was clean
4. No domain expert involvement

> DS team didn't fail individually—it was **collective oversight in scoping** and clarifying real business needs.

---

## Common Assumptions That Kill Projects

### 1. Assumption of Business Knowledge

**Business says**: "Products aren't sorted by our supplier agreements. We have contracts to abide by."

**ML team thinks**: "We have contracts? Nobody told us."

#### Case Study: Book Recommendation

- ML team built system recommending trending books
- Business requirement: Sponsored/contracted books should appear first
- Nobody told ML team about vendor relationships

**Fix**: Engage stakeholders early. Clarify requirements. Involve subject matter experts (e.g., vendor relationship manager).

---

### 2. Assumption of Data Quality

**Question**: "Why do we have duplicate products shown?"

**Answer**: Different product IDs, different SKUs, different descriptions → duplicates in recommendations.

| Reality | Statistic |
|---------|-----------|
| Struggling with data quality (prototype → production) | 68% |
| Struggling with data quality (production projects) | 75% |

**Lesson**: Data quality issues are common. Always vet data during early planning stages. Don't just take "data is clean" assertions at face value.

---

### 3. Assumption of Functionality

**Business**: "Why do my recommendations show a product I bought last week?"

**ML team**: "Oh, we didn't think of that. We can put it on the roadmap."

**Impact**: Recommending recently purchased items → destroys user trust.

**Fix**: 
- Take feedback as opportunity to improve
- Be transparent about cost of implementing new requirements
- Clarify user expectations during scoping

---

### 4. Curse of Knowledge

**ML team says**: "We found better results with hybrid matrix factorization and deep learning, but it adds 10 weeks."

**Business team thinks**: "What? Was that English?"

#### Example: Showing Gradient Descent Plot to Stakeholders

Don't show this:
- 3D gradient descent plots
- Local maxima/minima visualizations
- Complex mathematical metrics

**Instead**: 
- "We tested multiple approaches with varying impacts on time and performance"
- Let your audience guide how technical the discussion should get
- **Don't assume everyone has your knowledge**

---

### 5. Analysis Paralysis

**ML team**: "We spent 2 weeks on deep-and-wide AI approach but data feature consistency is so poor we can't proceed."

**Business team**: "Simple question—can we build this product or not?"

**Problem**: Overanalyzing every possible model/metric/parameter stalls progress.

> The goal is NOT always to build the **best** model. It's to build a **useful** one.

### Example: ChatGPT Releases

| Release | What They Did |
|---------|---------------|
| GPT-3.5 | Released early, collected human feedback (thumbs up/down) |
| GPT-4 | Improved based on feedback |
| GPT-4.1 | Further iterations |

**If they waited for the perfect model (GPT-4) before releasing anything**, competitors (Copilot, Claude) would have captured the market.

**Key insight**: 
- Release v1 that works
- Get feedback
- Iterate and improve
- **Progress = delivering solutions that work today, not obsessing over marginally better tomorrow**

---

## Quick Interview Points

1. **Poor planning/scoping** = #1 reason ML projects fail
2. **Paradigm shift**: From "how to build" → "what to solve and for whom"
3. **Cross-functional teams** (business + ML + design) = better outcomes
4. **Adequate planning** = structured research, feedback loops, iterative demos
5. **Data quality** = major struggle for 68-75% of projects
6. **Curse of knowledge** = explain in terms stakeholders understand
7. **Analysis paralysis** = ship useful, not perfect

---

## Recap

| Topic | Key Takeaway |
|-------|--------------|
| What fails ML projects | Poor planning and scoping, not technology |
| Paradigm shift | "What are we solving and for whom?" |
| Cross-functional planning | Business + ML + Design collaboration |
| Adequate vs inadequate | Structured research + feedback vs random experimentation |
| Common assumptions | Business knowledge, data quality, functionality, curse of knowledge |
| Analysis paralysis | Build useful models, iterate to perfect |

---

## Next Topics
- More on planning aspects
- Scoping in detail (Lecture 4)
