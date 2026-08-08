# OWL: Optimized Workforce Learning for General Multi-Agent Assistance for Real-World Task Automation

We present Workforce, a hierarchical multi-agent framework that decouples planning from execution through a modular
architecture with a domain-agnostic Planner, Coordinator, and specialized Workers. This enables cross-domain transfer by
allowing worker modification without full system retraining. On the GAIA benchmark, Workforce achieves state-of-the-art
69.70% accuracy, outperforming commercial systems.

This repository contains inference part code for the OWL framework (Workforce).

## Inference

The framework is based on `camel-0.2.46` version with minor modifications. To reproduce Workforce inference performance on GAIA benchmark (69.70% - Claude-3.7 accuracy on GAIA benchmark, pass@1, and 60.61% - GPT-4o accuracy on GAIA benchmark, pass@3), follow the steps below:

### Installation and Setup

1. Create a Python 3.11 Conda environment:

```bash
conda create -n owl python=3.11
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up envionment variables:

copy `.env.example` to `.env` and set the environment variables, and set the keys in `.env` file.

4. Run the inference:

- For reproducing results using GPT-4o, run:

```bash
python run_gaia_workforce.py
```

- For reproducing results using Claude-3.7, run:

```bash
python run_gaia_workforce_claude.py
```

You can modify `test_idx` variable to specify the test case.

