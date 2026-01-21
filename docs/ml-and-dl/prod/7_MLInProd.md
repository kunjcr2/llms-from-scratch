# Docker for ML Deployment

> Focus: Why Docker matters, containerization basics, and how to ship ML models that actually run on other machines.

---

## Core Problem

**"It works on my machine but not on yours."**

Your friend tries to run your ML code → version mismatch, missing libraries, config differences → hours of debugging.

Docker solves this by packaging **everything** (code, dependencies, config) into a portable container.

---

## What Is Docker?

| Concept | Explanation |
|---------|-------------|
| **Containerization** | Pack your app + dependencies into a standardized unit |
| **Container** | Running instance of an image — isolated, lightweight |
| **Image** | Static blueprint/template for a container |
| **Dockerfile** | Instructions to build an image |
| **Docker Hub** | Registry to store and share images |

> Real-life analogy: Shipping containers. Whatever's inside stays intact regardless of how it's transported — by ship, truck, or plane.

---

## Why Docker > "Just Share the Code"

| Without Docker | With Docker |
|----------------|-------------|
| "Install Python 3.9" → They have 3.11 | Same Python version guaranteed |
| "pip install sklearn" → They get different version | Exact versions from `requirements.txt` |
| Works on Mac, breaks on Linux | Runs identically everywhere |
| Hours debugging environment | `docker run` and done |

---

## Docker vs Virtual Machines

| Aspect | Docker Container | Virtual Machine |
|--------|------------------|-----------------|
| **OS** | Shares host kernel | Runs full guest OS |
| **Startup** | Seconds | Minutes |
| **Size** | MBs | GBs |
| **Performance** | Near-native | Overhead from virtualization |
| **Isolation** | Process-level | Full OS-level |
| **Security** | Good, but shares kernel | Stronger isolation |

> Docker = lightweight takeout box. VM = building a full restaurant.

---

## Key Docker Components

| Component | What It Does | Real-World Analogy |
|-----------|--------------|-------------------|
| **Docker Client** | CLI to issue commands | Your terminal |
| **Docker Daemon** | Background service managing everything | Hotel manager |
| **Docker Image** | Blueprint/template | Recipe card |
| **Docker Container** | Running instance | The actual cooked meal |
| **Dockerfile** | Build instructions | Ingredient list + steps |
| **Docker Registry** | Image storage (Docker Hub) | Recipe library |

---

## Docker Architecture Flow

```
Dockerfile → docker build → Image → docker run → Container
                                         ↓
                                   Docker Registry
                                   (share with team)
```

---

## Essential Docker Commands

### Check Installation

```bash
docker run hello-world
```

### Build an Image

```bash
docker build -t my-app-name .
```

- `-t` = tag (name your image)
- `.` = use Dockerfile in current directory

### Run a Container

```bash
docker run -p 5000:5000 my-app-name
```

- `-p host:container` = port mapping

### Save Image to Share

```bash
docker save -o my-app.tar my-app-name
```

### Load Shared Image

```bash
docker load -i my-app.tar
```

---

## Writing a Dockerfile

### Basic Structure

```dockerfile
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run command
CMD ["python", "app.py"]
```

### What Each Line Does

| Instruction | Purpose |
|-------------|---------|
| `FROM` | Base image (Python, Node, etc.) |
| `WORKDIR` | Create and set working directory |
| `COPY` | Copy files from host to container |
| `RUN` | Execute commands during build |
| `EXPOSE` | Document which port the app uses |
| `CMD` | Command to run when container starts |

---

## Practical Example: Deploying Iris Classifier

### Project Structure

```
iris-docker/
├── app.py           # Flask API
├── train.py         # Train and save model
├── requirements.txt # Dependencies
├── Dockerfile       # Container instructions
└── templates/
    └── index.html   # Web interface
```

### Workflow

1. **Train model locally**: `python train.py` → saves `model.pkl`
2. **Build image**: `docker build -t iris-docker-app .`
3. **Run container**: `docker run -p 5000:5000 iris-docker-app`
4. **Access**: `http://localhost:5000`
5. **Share**: `docker save -o iris-docker-app.tar iris-docker-app`

---

## Docker Desktop GUI

| Tab | What You See |
|-----|--------------|
| **Containers** | Running/stopped containers, start/stop controls |
| **Images** | Built images, size, creation time |
| **Volumes** | Persistent storage (survives container deletion) |

---

## Quick Interview Points

1. **Docker solves** → "Works on my machine" problem
2. **Container vs VM** → Shares OS kernel, faster startup, smaller footprint
3. **Image** → Blueprint/template (static)
4. **Container** → Running instance of image (dynamic)
5. **Dockerfile** → Build instructions for image
6. **Port mapping** → `-p 5000:5000` connects host to container
7. **Why MLOps uses Docker** → Reproducible environments, easy deployment, team collaboration

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `requirements.txt` | Always include with pinned versions |
| Not exposing ports | Add `EXPOSE` in Dockerfile + `-p` flag when running |
| Building without `.dockerignore` | Exclude `__pycache__`, `.git`, large files |
| Hardcoding paths | Use relative paths, `WORKDIR` in Dockerfile |

---

## Next Topics

- Flask for ML model serving
- End-to-end ML pipeline project
- Cloud deployment (AWS, GCP)
