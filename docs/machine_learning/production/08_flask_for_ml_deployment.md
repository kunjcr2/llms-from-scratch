# Flask for ML Deployment

> Focus: Flask fundamentals, modular coding practices, and building production-ready ML microservices.

---

## Why Flask for ML?

| Reason | What It Means |
|--------|---------------|
| **Lightweight** | No bloat — just what you need |
| **Fast API creation** | Spin up an endpoint in minutes |
| **Docker-friendly** | Easy containerization |
| **Separation of concerns** | Model logic stays isolated from UI |

> Flask is the go-to for serving ML models as APIs. It doesn't fight you — you write Python, Flask serves it.

---

## Core Flask Concepts

### 1. Application Setup

```python
from flask import Flask
app = Flask(__name__)
```

| Element | What It Does |
|---------|--------------|
| `Flask` | Main class — creates the app object |
| `__name__` | Tells Flask where to find templates/static files |

---

### 2. Routing (Endpoints)

```python
@app.route("/predict", methods=["POST"])
def predict():
    return "prediction result"
```

| Concept | Explanation |
|---------|-------------|
| `@app.route("/path")` | Maps URL to a function |
| `methods=["GET", "POST"]` | Which HTTP methods are allowed |
| Dynamic routes | `@app.route("/user/<username>")` → captures variable |

**Dynamic Route Example:**

```python
@app.route("/user/<username>")
def profile(username):
    return f"Hello, {username}"
```

---

### 3. Request & Response

```python
from flask import request, jsonify

@app.route("/api", methods=["POST"])
def api():
    data = request.json
    return jsonify({"received": data})
```

| Object | Purpose |
|--------|---------|
| `request` | Access incoming data (headers, form, JSON, files) |
| `jsonify()` | Return JSON response |

---

### 4. Sessions & Cookies

```python
from flask import session
session["user"] = "Prathamesh"
```

- `session` → Secure, server-side dict stored in signed cookies
- Useful for storing user state between requests

---

### 5. Global Object (`g`)

- `g` → Temporary storage during a single request lifecycle
- Common use: DB connections, request-specific data

---

## Jinja2 Templating

> Flask uses Jinja2 for rendering dynamic HTML.

### Syntax

| Syntax | Purpose |
|--------|---------|
| `{{ variable }}` | Output a value |
| `{% ... %}` | Control structures (loops, conditionals) |
| `{# ... #}` | Comments |

### Example

```html
<h1>Welcome {{ user }}</h1>
{% if items %}
<ul>
    {% for i in items %}
    <li>{{ i }}</li>
    {% endfor %}
</ul>
{% else %}
<p>No items available</p>
{% endif %}
```

---

## Request Handling Lifecycle

```
1. Request     → Client sends HTTP request
2. Routing     → Flask matches URL to function
3. View Func   → Your Python code runs
4. Response    → Flask returns HTML/JSON/file
5. Middleware  → Extensions can intercept at any stage
```

> This matters in interviews — shows you understand what happens under the hood.

---

## Modular Coding Practices

### Why Modular?

| Without Modular Code | With Modular Code |
|---------------------|-------------------|
| One giant script | Separate files for each concern |
| Hard to test | Easy unit testing |
| "Works on my machine" issues | Reproducible structure |
| No reusability | Reusable components |

### Typical ML Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── data_loader.py
│   ├── models/
│   │   └── classifier.py
│   ├── utils/
│   │   └── helpers.py
│   └── api/
│       └── app.py
├── config/
│   └── config.yaml
├── tests/
│   └── test_classifier.py
├── Dockerfile
├── requirements.txt
└── README.md
```

### Key Principles

| Principle | What It Means |
|-----------|---------------|
| **Single Responsibility** | Each module does one thing |
| **Config Separation** | Params in config files, not hardcoded |
| **Testability** | Functions should be unit-testable |
| **Reusability** | Write functions you can use across projects |

---

## Flask + ML Integration Pattern

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model once at startup
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### Why This Pattern Works

- Model loads **once** (not per request)
- API is stateless — easy to scale
- JSON in, JSON out — standard REST

---

## Flask Strengths & Limitations

### Strengths

| ✅ Pro | Why It Matters |
|-------|----------------|
| Lightweight | No unnecessary overhead |
| Pythonic | Explicit is better than implicit |
| Large extension ecosystem | Auth, DB, etc. available |
| Easy learning curve | Quick to prototype |

### Limitations

| ❌ Con | What It Means |
|--------|---------------|
| No built-in admin panel | Unlike Django — you build your own |
| Not opinionated | More freedom = potential inconsistency |
| Scalability = your problem | Design matters — Flask won't save you |

---

## Quick Interview Points

1. **Why Flask for ML?** → Lightweight, easy API creation, Docker-friendly
2. **Route** → `@app.route()` maps URL to function
3. **request object** → Access incoming JSON/form/files
4. **jsonify()** → Return JSON response
5. **Jinja2** → Templating engine for dynamic HTML
6. **Modular code** → Separate concerns, easier testing, reproducibility
7. **Model loading pattern** → Load once at startup, not per request
8. **Flask vs Django** → Flask is minimal; Django is batteries-included

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Loading model in every request | Load once at app startup |
| Hardcoding config values | Use config files or env vars |
| No error handling in endpoints | Wrap in try/except, return proper status codes |
| Forgetting CORS for frontend | Add `flask-cors` extension |
| Not setting `host="0.0.0.0"` | Required for Docker to expose properly |

---

## What's Coming Next (Part 2)

- Hands-on Resume Screening Project
- Full Flask + ML + Docker integration
- End-to-end deployment workflow
