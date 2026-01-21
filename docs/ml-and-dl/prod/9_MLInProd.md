# CI/CD for ML Pipelines

> Focus: Why CI/CD matters, the pipeline architecture, and how automation transforms ML deployment from days to minutes.

---

## The Core Problem

**Without CI/CD:**
- Developer A and Developer B work on separate branches
- They don't know what each other changed
- Merge day = chaos (conflicts, broken builds, failed tests)
- Releasing takes days/weeks
- Bugs found late = expensive fixes

**With CI/CD:**
- Everyone pushes to shared repo frequently
- Automated tests catch issues immediately
- Releases happen in minutes
- Netflix can update recommendations same day a movie trends

> Analogy: Google Docs vs Word files. Everyone editing the same doc in real-time vs. merging 10 separate .docx files the night before deadline.

---

## What Is CI/CD?

| Term | Full Name | What It Does |
|------|-----------|--------------|
| **CI** | Continuous Integration | Frequently merge code → auto build → auto test |
| **CD** | Continuous Delivery | Code is always deployable (one-click manual deploy) |
| **CD** | Continuous Deployment | Auto-deploy every validated build to production |

> CI/CD = automate building, testing, and deploying code changes.

---

## Pre-CI/CD Era Problems

| Problem | Why It Hurt |
|---------|-------------|
| **Merge conflicts** | Multiple devs changing same code unknowingly |
| **Broken builds** | Code worked locally, dies when combined |
| **Late bug detection** | Testing only near release = expensive fixes |
| **Manual deployments** | Slow, error-prone, someone always forgets a step |
| **Poor collaboration** | Dev, test, ops teams working in silos |

---

## The Restaurant Analogy

| Role | CI/CD Equivalent |
|------|------------------|
| **Chef** | Continuous Integration — cooks (builds) and tastes (tests) the food |
| **Waiter** | Continuous Delivery — takes food to customer (manual step) |
| **Self-service/Cloud kitchen** | Continuous Deployment — food reaches customer automatically |

---

## CI/CD Pipeline Flow

```
Plan → Code → Build → Test → Release → Deploy → Operate → Monitor
      |______________|      |__________________________|
            CI                        CD
```

### Step Breakdown

| Stage | What Happens | Tools |
|-------|--------------|-------|
| **Plan** | Define tasks, track issues | Jira |
| **Code** | Write and commit code | Git, GitHub, GitLab |
| **Build** | Compile, create artifacts | Jenkins, GitHub Actions |
| **Test** | Run automated tests | pytest, Jest, Selenium |
| **Release** | Prepare for deployment | Docker, Helm |
| **Deploy** | Push to staging/production | AWS, GCP, Azure |
| **Operate** | Run in production | Kubernetes, Docker |
| **Monitor** | Track performance, errors | Splunk, Sumo Logic, Prometheus |

---

## Continuous Integration (CI)

### What Happens

1. Developer commits code
2. Triggers automated build
3. Runs unit + integration tests
4. Static code analysis
5. Quality checks

### Objectives

- Detect issues **early** (not release day)
- Keep codebase **stable and deployable**
- Reduce merge conflicts
- Enable collaboration

### Popular CI Tools

| Tool | Notes |
|------|-------|
| **GitHub Actions** | Built into GitHub, easy setup |
| **Jenkins** | Most popular, highly customizable |
| **GitLab CI** | Integrated with GitLab |
| **CircleCI** | Cloud-native, fast |

---

## Continuous Delivery vs Deployment

| Aspect | Continuous Delivery | Continuous Deployment |
|--------|---------------------|----------------------|
| **Deploy trigger** | Manual button click | Automatic |
| **Human approval** | Required | None |
| **Use case** | Weekly/monthly releases | Real-time updates |
| **Risk** | Lower (human review) | Higher (no human gate) |

### When to Use What

- **Continuous Delivery** → You're okay releasing weekly, need human approval for compliance
- **Continuous Deployment** → Amazon homepage changes hourly, Netflix recommendations update instantly

---

## DevSecOps (Security in CI/CD)

> Modern pipelines integrate security **early** — not after deployment.

### Shift Left Security

| Traditional | Shift Left |
|-------------|------------|
| Check security after building | Check security from the start |
| Find vulnerabilities in production | Catch them in code review |
| Expensive fixes | Cheap fixes |

> Analogy: Check foundation strength **while building**, not after the 100-floor skyscraper is complete.

### Security Measures in Pipeline

| Measure | What It Does |
|---------|--------------|
| **SAST** (Static Application Security Testing) | Detects vulnerabilities in source code |
| **Dependency Scanning** | Checks if third-party libraries are vulnerable |
| **Container Scanning** | Verifies Docker images are secure |
| **Secret Management** | Prevents accidental API key/credential exposure |

---

## Real-World Example: Password Policy Change

**Scenario:** You update password rules (10 chars, 3 uppercase, 2 lowercase, 1 symbol)

**Without proper CI/CD:**
- Old users can't log in (their passwords don't meet new rules)
- No one tested backward compatibility
- Production breaks

**With proper CI/CD:**
- Test cases catch this edge case
- Build fails before merge
- Force dev to add: "Apply new rules for new users only; existing users migrate on password reset"

---

## Benefits of CI/CD

| Benefit | Why It Matters |
|---------|----------------|
| **Fast integration** | Hours, not days |
| **Fewer merge conflicts** | Small, frequent commits |
| **Early bug detection** | Fix when it's cheap |
| **Consistent test environment** | Same tests every time |
| **Team collaboration** | Dev, QA, Ops work together |
| **Reliable releases** | Automated = no forgotten steps |

---

## Quick Interview Points

1. **CI** → Merge frequently, auto build, auto test
2. **CD (Delivery)** → Always deployable, one-click deploy
3. **CD (Deployment)** → Auto-deploy every passing build
4. **Why CI/CD?** → Faster releases, fewer bugs, better collaboration
5. **Shift left** → Move security checks earlier in pipeline
6. **Jenkins vs GitHub Actions** → Jenkins = customizable, mature; GH Actions = simple, integrated
7. **DevSecOps** → Security baked into pipeline, not bolted on later

---

## Common Interview Questions

| Question | Key Points to Hit |
|----------|-------------------|
| "What's the difference between CI and CD?" | CI = integrate + test; CD = deploy (manual or auto) |
| "Why is CI/CD important for ML?" | Models change, data changes, need frequent retraining + deployment |
| "What tools have you used?" | Jenkins, GitHub Actions, GitLab CI — pick one you know |
| "What is shift-left security?" | Move security checks earlier; catch issues in dev, not prod |
| "How do you handle failed builds?" | Block merge, notify dev, fix before proceeding |

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Tests take too long | Parallelize, prioritize critical tests |
| No tests at all | At minimum: unit tests for core logic |
| Hardcoded secrets in code | Use secret management (Vault, env vars) |
| Skipping staging environment | Always test in staging before prod |
| No rollback plan | Automate rollback on failure |

---

## What's Coming Next (Part 2)

- Practical CI/CD implementation
- GitHub Actions or Jenkins hands-on
- Anomaly detection project with full pipeline
