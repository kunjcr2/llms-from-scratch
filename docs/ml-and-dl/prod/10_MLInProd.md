# Deployment & Monitoring for ML

> Focus: Why deployment matters, platform comparison, AWS EC2 hands-on, and connecting GitHub CI/CD to cloud deployment.

---

## Why Deployment Matters

**The harsh truth:**
- 99% accuracy in Jupyter notebook = 0% business value
- Model sitting locally = no users, no feedback, no impact

> Analogy: Cooking amazing food at home vs. opening a restaurant. Your friends might say it's great, but until strangers pay for it and leave Google reviews, you don't really know.

### What Deployment Enables

| Before Deployment | After Deployment |
|-------------------|------------------|
| Runs on your laptop | Runs 24/7 on cloud |
| You test it manually | Real users generate feedback |
| No scale | Handles thousands of requests |
| "It works for me" | Works for everyone |

---

## Why Monitoring Is Critical

> Deployment is not the finish line. It's where the real problems start.

### What Can Go Wrong

| Issue | Example |
|-------|---------|
| **Data drift** | New restaurants added daily — model needs fresh data |
| **System failure** | API crashes at 3 AM — no one notices until morning |
| **Performance degradation** | Model accuracy drops slowly over time |
| **Latency spikes** | Search takes 10 seconds instead of 1 |

### Three Levels of Monitoring

| Level | What to Monitor | Examples |
|-------|-----------------|----------|
| **System** | Infrastructure health | CPU usage, memory, disk, network, API latency |
| **Data** | Input quality | Missing values, distribution shifts, invalid data |
| **Model** | Prediction quality | Accuracy, F1 score, reconstruction error |

---

## Deployment Platforms Comparison

### AWS EC2 (Elastic Compute Cloud)

**What it is:** Virtual machine you fully control — like having your own remote computer.

**Architecture:**

```
Region (e.g., Asia Pacific)
└── VPC (Virtual Private Cloud)
    └── Availability Zone (physical data center)
        └── Public Subnet (internet access)
            └── Security Group (firewall — opens specific ports)
                └── EC2 Instance (your virtual machine)
                    ├── AMI (OS image)
                    └── EBS (storage volume)
```

| Component | What It Does |
|-----------|--------------|
| **Region** | Physical location of servers |
| **VPC** | Your private network space |
| **Security Group** | Firewall — define which ports are open |
| **AMI** | Pre-configured OS image (Ubuntu, Amazon Linux, etc.) |
| **EBS** | Storage attached to your instance |

**Pros:**
- Full control over OS, dependencies, configurations
- Ideal for custom ML pipelines
- Easy Docker + CI/CD integration
- Good for research/demo projects

**Cons:**
- Manual scaling
- You manage security, monitoring, updates
- Requires sysadmin knowledge

---

### Google Cloud Run

**What it is:** Serverless container platform — no servers to manage.

**Pros:**
- No infrastructure management
- Automatic scaling (scales to zero when idle)
- Pay only for what you use (billing by request)
- Great for lightweight ML APIs

**Cons:**
- No control over infrastructure
- Cold start latency (first request after idle is slow)
- Less suitable for long-running ML jobs

---

### Microsoft Azure

**What it is:** Enterprise-grade cloud with full ML pipeline support.

**Architecture Flow:**

```
Unstructured/Structured Data
    ↓
Azure Synapse Analytics (ingest + preprocess)
    ↓
Azure Data Lake Storage (store raw/processed data)
    ↓
Apache Spark (feature engineering + model training)
    ↓
Cosmos DB (store predictions)
    ↓
Power BI / Web App (dashboard + user interface)
```

**Pros:**
- Strong enterprise integration
- Managed ML services
- Excellent monitoring/logging tools

**Cons:**
- Steep learning curve
- Higher cost for small projects
- Configuration complexity

---

### Render

**What it is:** Simple deployment platform for beginners.

**Pros:**
- Dead simple — connect GitHub repo, deploy
- Minimal configuration
- Great for quick demos and prototypes

**Cons:**
- Limited control over scaling
- Not for heavy ML workloads
- Limited customization

---

## Platform Decision Matrix

| Platform | Best For | Control | Scaling | Cost Efficiency |
|----------|----------|---------|---------|-----------------|
| **AWS EC2** | Research, CI/CD, custom pipelines | High | Manual | Medium |
| **Google Cloud Run** | Lightweight APIs, variable traffic | Low | Auto | High |
| **Azure** | Enterprise systems | Medium | Managed | Low (for small projects) |
| **Render** | Prototypes, demos | Low | Auto | High |

### Quick Decision Guide

- **Need full control?** → AWS EC2
- **Stateless API with variable traffic?** → Google Cloud Run
- **Enterprise-grade with compliance needs?** → Azure
- **Just want to demo quickly?** → Render

---

## AWS EC2 Hands-On Setup

### 1. Launch Instance

1. Go to EC2 Dashboard → Launch Instance
2. Choose **Ubuntu Server 24.04** (free tier eligible)
3. Select instance type: **t2.micro** (free tier) or **t3.micro**
4. Create key pair (RSA, `.pem` file) — **save this file safely**
5. Configure Security Group:
   - Port 22 (SSH)
   - Port 80 (HTTP)
   - Port 443 (HTTPS)
   - Port 8000 (FastAPI)
6. Set storage to **20 GB** (8 GB default is often insufficient for Docker)

### 2. Connect via SSH

```bash
ssh -i /path/to/your-key.pem ubuntu@<your-public-ipv4>
```

### 3. Install Docker on EC2

```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

---

## Connecting GitHub Actions to EC2

### Add GitHub Secrets

Go to: **Repository → Settings → Secrets and variables → Actions**

| Secret Name | Value |
|-------------|-------|
| `EC2_HOST` | Your public IPv4 address (no http://) |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | Contents of your `.pem` file |
| `GHCR_PAT` | Personal Access Token (for GitHub Container Registry) |

### Create Personal Access Token

1. GitHub → Settings → Developer Settings → Personal Access Tokens → Classic
2. Generate new token with scopes:
   - `repo` (full access)
   - `workflow`
   - `write:packages`
3. Copy and save as `GHCR_PAT` secret

### CD Workflow (deploy to EC2)

```yaml
name: Deploy to AWS EC2

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    
    steps:
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_SSH_KEY }}
          port: 22
          script: |
            echo ${{ secrets.GHCR_PAT }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
            cd ${{ secrets.DOCKER_COMPOSE_DIR }}
            docker-compose pull
            docker-compose down
            docker-compose up -d
```

---

## Common Gotchas & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| **SSH handshake failed** | Extra whitespace in `.pem` key | Use `pbcopy < key.pem` instead of manual copy-paste |
| **docker compose not found** | Docker Compose v1 vs v2 syntax | Use `docker-compose` (v1) or `docker compose` (v2) |
| **Storage full** | Default 8GB insufficient for Docker | Increase to 20-30 GB when launching |
| **Connection refused** | Port not open in Security Group | Add inbound rule for the port (8000, 80, etc.) |
| **IPv4 changed after stop/start** | EC2 assigns new IP on restart | Use Elastic IP for persistent address |

---

## Quick Interview Points

1. **Why deploy ML models?** → Notebook accuracy means nothing without real users
2. **Why monitor?** → Data drifts, systems fail, performance degrades silently
3. **EC2 vs Cloud Run** → EC2 = control + manual scaling; Cloud Run = serverless + auto-scaling
4. **Three monitoring levels** → System (CPU/memory), Data (distribution), Model (accuracy)
5. **Security Group** → Firewall that controls which ports accept traffic
6. **AMI** → Pre-configured OS image for quick instance setup
7. **Cold start** → Delay when serverless function wakes from idle

---

## Production Checklist

| ✅ | Item |
|----|------|
| | Model loads once at startup (not per request) |
| | Health check endpoint exists |
| | Logging configured (requests, errors, predictions) |
| | Monitoring dashboards set up |
| | Alerts for latency spikes / error rates |
| | Rollback plan documented |
| | Security groups configured correctly |
| | Secrets stored in environment variables / vault |

---

## What's Next

- **Mega Project**: End-to-end ML pipeline
  - Data scraping from multiple sources
  - MongoDB integration
  - Flask + Docker + AWS EC2
  - Full CI/CD pipeline
  - Production dashboard
