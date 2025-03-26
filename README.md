# Jo.E: Joint Evaluation Framework

This repository contains the implementation of the Joint Evaluation (Jo.E) framework as described in the paper [Joint Evaluation (Jo.E): A Collaborative Framework for Rigorous Safety and Alignment Evaluation of AI Systems Integrating Human Expertise, LLMs, and AI Agents][https://github.com/HimJoe/jo-e-framework/blob/main/Joint%20Evaluation%20(Jo.E).pdf)]
<img width="1004" alt="Screenshot 2025-03-25 at 8 58 23 PM" src="https://github.com/user-attachments/assets/35e80ae9-fbde-4d07-962d-8d81645bd5b4" />

## Overview

Jo.E is a structured evaluation framework that integrates:
- Human expertise
- AI agents
- Large Language Models (LLMs)

To systematically assess AI systems across critical dimensions:
- Accuracy
- Robustness
- Fairness
- Ethical compliance

## Repository Structure

- `framework/`: Core Jo.E framework components
  - `llm_evaluators/`: LLM-based initial evaluators
  - `ai_agents/`: Specialized AI agents for systematic testing
  - `human_evaluation/`: Human expert review interfaces and protocols
  - `scoring/`: Implementation of the comprehensive scoring system

- `experiments/`: Reproduces the experimental results from the paper
  - `experiment1_model_performance/`: Comparative model performance analysis
  - `experiment2_domain_specific/`: Domain-specific evaluation (legal and customer service)
  - `experiment3_adversarial/`: Adversarial robustness testing

- `streamlit_app/`: Interactive web application for using the Jo.E framework
- `datasets/`: Sample datasets for running experiments
- `docs/`: Additional documentation and resources

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/jo-e-framework.git
cd jo-e-framework
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

### Running Experiments

```bash
cd experiments/experiment1_model_performance
python run_experiment.py
```

## Citation

If you use this framework in your research, please cite:

```
@article{joshi2025joint,
  title={Joint Evaluation (Jo.E): A Collaborative Framework for Rigorous Safety and Alignment Evaluation of AI Systems Integrating Human Expertise, LLMs, and AI Agents},
  author={Joshi, Himanshu},
  journal={Vector Institute for Artificial Intelligence},
  year={2025}
}
```

## License

MIT License
