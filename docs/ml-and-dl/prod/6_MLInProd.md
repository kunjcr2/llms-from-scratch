# How to Write Modular ML Code

> Focus: Breaking monolithic scripts into clean, testable, production-ready modules. Folder structure, config files, and TDD for ML.

---

## Core Insight

**Monolithic notebooks get the job done in experiments but are impossible to maintain, test, or deploy in production.**

The goal: write code like building with Lego blocks—each piece does one thing, can be replaced, and snaps together cleanly.

---

## Why Monolithic Scripts Are a Production Killer

| Problem | Impact |
|---------|--------|
| **Hard to debug** | Wall of text → finding a missing comma is like finding a needle in a haystack |
| **Not reusable** | Can't extract the model architecture for a new dataset without copy-pasting |
| **Not testable** | No unit tests, no CI/CD possible |
| **Not scalable** | Everything breaks when you try to add features |
| **Collaboration nightmare** | Multiple people can't work on different parts |

> Real-life analogy: Cooking 8 dishes in one bowl. It works, but it's chaotic and unsustainable.

---

## The Modular Approach

### What It Means

Break your ML pipeline into **smaller, independent, testable pieces**:

| Module | Responsibility |
|--------|----------------|
| `data_loader.py` | Load and preprocess data |
| `model.py` | Define model architecture |
| `train.py` | Training loop, save model |
| `evaluate.py` | Metrics, validation |
| `config.yaml` | Hyperparameters, paths |
| `main.py` | Orchestrate everything |

### Why It Matters

- **Single Responsibility**: Each file does one thing well
- **Loose Coupling**: Change one module without breaking others
- **Testable**: Write unit tests for each component
- **Reusable**: Use the same training loop for different models

---

## Recommended Folder Structure

```
project_name/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── config/
│   └── config.yaml
├── main.py
├── requirements.txt
├── setup.py
└── README.md
```

> Interview signal: If you can describe this structure and explain *why* each piece exists, you sound like someone who ships code.

---

## Configuration Files (YAML/JSON)

### Why Use Them

| Benefit | Example |
|---------|---------|
| **Change params without touching code** | `n_estimators: 100 → 200` |
| **Experiment tracking** | Each run's config is a record |
| **Reproducibility** | Same config = same results |
| **Team collaboration** | Share configs, not code diffs |

### Example `config.yaml`

```yaml
data:
  dataset: iris
  
model:
  type: random_forest
  n_estimators: 100
  max_depth: 4

train:
  test_size: 0.2
  random_state: 42

output:
  model_path: model.joblib
```

### Loading Config in Python

```python
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

---

## Test-Driven Development (TDD) for ML

### Why It's Rare But Effective

- **Catches bugs early**: Define what "correct" looks like before coding
- **Safe refactoring**: Modify code confidently because tests validate behavior
- **Documents intent**: Tests show how code should be used

### Example: Testing Data Loader

```python
def test_load_data():
    X, y = load_data()
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 4  # Expected features
```

> Interview point: Mentioning TDD for ML shows you think about quality, not just accuracy.

---

## `requirements.txt` — Dependency Management

### Why It Matters

- Your code runs on your machine ≠ runs on everyone's machine
- Different library versions = different bugs

### Example

```
scikit-learn
pandas
numpy
pyyaml
joblib
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Practical Example: Modular Iris Classifier

### `src/data_loader.py`

```python
from sklearn.datasets import load_iris

def load_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return X, y
```

### `src/model.py`

```python
from sklearn.ensemble import RandomForestClassifier

def build_model(config):
    params = config['model']
    return RandomForestClassifier(**params)
```

### `src/train.py`

```python
from sklearn.model_selection import train_test_split
import joblib

def train(X, y, model, config):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['train']['test_size'],
        random_state=config['train']['random_state']
    )
    model.fit(X_train, y_train)
    joblib.dump(model, config['output']['model_path'])
    return model, X_test, y_test
```

### `src/evaluate.py`

```python
from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
```

### `main.py`

```python
import yaml
from src import data_loader, model, train, evaluate

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    X, y = data_loader.load_data()
    clf = model.build_model(config)
    clf, X_test, y_test = train.train(X, y, clf, config)
    evaluate.evaluate(clf, X_test, y_test)
```

### Run It

```bash
python main.py
# Output: Accuracy: 1.0
```

---

## Quick Interview Points

1. **Monolithic = bad** → Hard to debug, test, reuse, scale
2. **Modular = production-ready** → Each file does one thing well
3. **Config files** → Change hyperparameters without touching code
4. **requirements.txt** → Version control for dependencies
5. **TDD in ML** → Define expected behavior before implementation
6. **Folder structure** → Shows you've shipped production code
7. **`main.py` as orchestrator** → Clean entry point, calls modules in order

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| All logic in one notebook cell | Split into separate `.py` files |
| Hardcoded paths/hyperparameters | Use config files |
| No `requirements.txt` | Generate with `pip freeze > requirements.txt` |
| Copy-pasting code between projects | Write reusable modules |

---

## Next Topics

- Complex modular pipelines
- Deployment workflows
- AWS SageMaker practical
