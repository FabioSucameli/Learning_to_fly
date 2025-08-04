# Learning To Fly

## Setup

This guide provides the minimal steps to set up and activate the Python environment for using the `gym-pybullet-drones` simulator on **Windows** using **Anaconda**.

---

## Prerequisites

- Python ≥ 3.10
- [Anaconda or Miniconda](https://www.anaconda.com/download) installed
- [Git for Windows](https://git-scm.com/) (optional, or download the ZIP manually)

---

## Installation Steps

### 1. Clone or Download the Project

**Option 1: Using Git**
```bash
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
```

**Option 2: Manual Download**
- Download the ZIP from: https://github.com/utiasDSL/gym-pybullet-drones
- Extract it
- Open Anaconda Prompt and navigate to the extracted folder:
```bash
cd path\to\gym-pybullet-drones
```

---

### 2. Create the Conda Environment

```bash
conda create -n drones python=3.10 -y
conda activate drones
```

---

### 3. Install Dependencies

```bash
pip install -e .
```

---


##  Notes

- For GUI rendering, make sure your system supports OpenGL and GPU drivers are up to date.
- If using WSL, GUI rendering might not work reliably — run on Windows directly for best results.

