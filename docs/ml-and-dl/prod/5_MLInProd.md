# Communication and Logistics of ML Projects

> Focus: How communication drives project success. Planning meetings, stakeholder alignment, and pre-production readiness.

---

## Core Insight

**Most ML projects fail not because of bad code, but because teams lacked shared understanding of what was being built and why.**

Focus should be on **what** and **why** before **how**.

---

## Why DS Projects Struggle to Gain Adoption

| Problem | Example |
|---------|---------|
| **Misaligned Language** | You: "ROC is 0.87" → Stakeholder: "Show me the top 5 trending movies" |
| **Overcommunicating Technical Depth** | You: "Dropout = 0.3, used collaborative filtering" → Stakeholder: "Will customers watch at least 2 movies?" |
| **Lack of Shared Problem Definition** | You: "Top 5 trending globally" → Stakeholder: "I want youth-targeted trending content for India" |
| **Assumed Shared Context** | You know data gaps exist → Business expects real-time updates |

> **Fix**: Communicate in business terms. Define success together. Share constraints early.

---

## Critical Questions to Ask in First Planning Meeting

| Question | What You Learn |
|----------|----------------|
| **Why do you want to build this?** | Urgency, importance, motivation |
| **What do you expect the solution to do?** | High-level scope, basic functionality |
| **How does your team do this now?** | Current process, subject matter experts, critical features |
| **What would be the perfect solution?** | Accuracy expectations, realistic vs ideal targets |
| **How much would you pay another company to do this?** | Seriousness, sophistication required |
| **When would this solution become irrelevant?** | Deadline constraints, scope creep prevention |

> These questions shift conversation from "how to build" to "what to build and why it matters."

---

## Guided vs Unguided Planning Meetings

### Unguided (Everyone Asks Different Questions)

| Role | Question |
|------|----------|
| Sponsor | "How does this increase sales?" |
| Business Lead | "Will retention go up?" |
| Project Manager | "How many sprints?" |
| DS Lead | "What GPU budget do we have?" |
| Data Scientist | "LLM or collaborative filtering?" |
| Frontend Dev | "What data structure will I receive?" |
| Data Engineer | "Activity in data lake, purchases in warehouse—how to join?" |

**Result**: No alignment. Project drifts.

### Guided (Team Discusses Core Questions Together)

1. **Why are we building this?**
2. **What do we want it to do?**
3. **How could this go wrong?**
4. **When do we want this done?**

**Result**: Shared understanding → Aligned decisions.

---

## High-Level Flowchart Diagrams

### Why They Matter

- Ground team in **problem space** before **solution space**
- Detect risks early
- Enable shared understanding between technical and non-technical teams

### What to Include

| Component | Example |
|-----------|---------|
| User actions | Login, product browsing |
| Backend interactions | Query user profile |
| Model outputs | Recommended items |
| Fallback systems | Cold start logic |
| UI integration | What user sees |

### Example: Recommendation System Flow

```
User Login → Web Server → User ID Query
                ↓
     ┌─────────────────────────────┐
     │ Personalized Item List      │
     │ + Cold Start/Fallback Items │
     │ + Global Prioritized Items  │
     └─────────────────────────────┘
                ↓
         Merged Output → Display Top 7 Items
```

> Stakeholders understand flowcharts. They don't understand architecture diagrams with gradients and embeddings.

---

## Simple → Detailed Diagrams

### Start Simple (For Stakeholders)

Show core flow with minimal boxes.

### Add "Nice to Have" Comments Later

| Feature | Type |
|---------|------|
| Remove similar items from recommendations if P1 added to cart | Nice to have |
| Similar filtering based on session activity | Nice to have |
| "Frequently bought together" suggestions | Nice to have |
| Save session history for returning users | Nice to have |

> Focus first on **functional MVP**. Revisit "nice to haves" once core system is stable.

---

## Who Is Your Champion?

**Not the project owner** (too busy, managing multiple projects).

**It's the Subject Matter Expert (SME):**
- Deep connection to the problem
- Can check work and answer questions
- Provides creative ideation
- Shapes project direction

> The right champion = clarity, speed, smarter decisions.

---

## Milestone-Based Meetings

### What They Are

Project meetings that coincide with **milestones** (not daily standups).

### Rules

| Do | Don't |
|----|-------|
| Align with project milestones | Substitute for daily standups |
| Have full team present | Hold individual meetings separately |
| Present solution as it stands | Deviate from main topic |
| Have project lead to decide contentious topics | Leave decisions hanging |

### Gantt Charts for Timeline Visualization

```
Week 1: Kickoff + Data Collection
Week 2: Model Building
Week 3-4: SME Review + Iteration
Week 5: Deployment
```

> Gantt charts: Visual project planning → clarity on who does what and when.

---

## Spotify Case Study: Discover Weekly

### Business Ask

"Can we personalize music for every user every week automatically?"

### Goal

Increase user engagement and retention through custom Monday playlists.

### Approach

| Phase | Action |
|-------|--------|
| Ideation | Product, DS, music experts, UI designers brainstorm |
| Discovery | Behavior graphs, collaborative filtering, NLP on metadata, audio CNN |
| MVP Focus | Recommend 30 songs/week based on listening + similar users + audio features (tempo, mood, pitch) |

### Excluded from MVP

- Social media buzz integration
- New artist promotion
- Live concert recommendations

### Result (2015 Launch)

- 40M+ users by 2016
- Billions of streams
- Flagship feature
- Customer retention achieved

> **Key**: Listen broadly, experiment wisely, focus sharply on one goal at a time.

---

## Pre-Production Readiness

### Code Complete ≠ Project Complete

Just like a drama script written + rehearsed ≠ ready for annual function.

**SMEs must validate output in real environment and judge solution qualitatively.**

### User Acceptance Testing (UAT) > Metrics Alone

Human judgment catches what metrics miss:
- Edge cases
- User experience issues
- Business logic gaps

### Pre-Production Review Checklist

| Item | Status |
|------|--------|
| Changes reviewed | ✓ |
| Performance verified | ✓ |
| Load simulated | ✓ |
| Analytics dashboard accessible | ✓ |
| Non-tech teams can access AB data | ✓ |

---

## Key Takeaways

| Principle | Summary |
|-----------|---------|
| **Plan with purpose** | Meet weekly only if there's agenda/milestone |
| **Human input > metrics** | UAT catches what numbers miss |
| **Prototype, trust real feedback** | Validate with experimentation + user acceptance |
| **Verify with SMEs** | Ensure alignment with business requirements |
| **Build simplest solution first** | Complexity can be added later; simplicity ships faster |

---

## Quick Interview Points

1. **Misaligned language** = Speak business, not data science lingo
2. **Shared problem definition** = Agree on success criteria upfront
3. **Guided planning** = Use the 4 core questions (why, what, how could it fail, when)
4. **Flowcharts > architecture diagrams** = For stakeholder communication
5. **Champion = SME** = Your go-to for domain clarity
6. **Milestone meetings** = Not daily standups; validate progress at checkpoints
7. **MVP first** = Validate before adding "nice to haves"
8. **UAT** = Human validation before launch
9. **Gantt charts** = Timeline visualization for planning

---

## Next Topics

- Critical decision boundaries
- User Acceptance Testing (UAT) deep dive
- AWS SageMaker practical (coding phase)
