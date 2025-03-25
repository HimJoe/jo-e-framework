# Jo.E Framework Streamlit App

This Streamlit application provides a user interface for the Joint Evaluation (Jo.E) framework,
allowing users to evaluate AI systems using the tiered evaluation pipeline with LLMs,
AI agents, and human expertise.

## Features

- **Interactive Model Evaluation**: Evaluate individual prompt-response pairs
- **Batch Evaluation**: Evaluate multiple examples using sample or custom datasets
- **Results Visualization**: View evaluation results with detailed breakdowns and visualizations
- **Human Review Interface**: Facilitate expert review of AI system outputs
- **Model Comparison**: Compare multiple AI systems across evaluation dimensions

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Other dependencies as listed in the main `requirements.txt` file

### Running the App

```bash
cd streamlit_app
streamlit run app.py
```

### Configuration

The app requires API keys for access to various LLM services:

- OpenAI API key (for GPT-4o evaluator)
- Anthropic API key (optional, for Claude evaluator)
- Llama API key (optional, for Llama evaluator)

These can be entered in the app interface or stored in environment variables.

## Usage Guide

### Step 1: Initialize the Framework

1. Enter API keys in the sidebar
2. Select which LLM evaluators and AI agents to use
3. Configure evaluation settings
4. Click "Initialize Framework"

### Step 2: Evaluate Models

#### Interactive Mode
- Enter a prompt and the model's response
- Click "Evaluate" to run the Jo.E evaluation pipeline

#### Batch Mode
- Select a sample dataset or upload your own JSON file
- Generate responses (or use pre-generated responses)
- Run batch evaluation

### Step 3: View Results

- Examine scores across dimensions (accuracy, robustness, fairness, ethics)
- View detailed results from each stage of the pipeline
- Export results as JSON for external analysis

### Step 4: Compare Models

- Select multiple models to compare
- Visualize relative performance with comparative charts
- Identify strengths and weaknesses of each model

## Implementation Details

The app integrates with the Jo.E framework components:

- **LLM Evaluators**: GPT-4o, Llama 3.2
- **AI Agents**: Adversarial, Bias Detection, Knowledge Verification, Ethical Boundary
- **Human Review**: Interactive interface for expert evaluation
- **Scoring System**: Multi-dimensional scoring across key aspects

## Customization

The app can be extended with:

- Additional LLM evaluators
- Specialized AI agents for specific domains
- Custom datasets for domain-specific evaluation
- Enhanced visualization options

## Related Resources

- [Jo.E Framework GitHub Repository](https://github.com/your-username/jo-e-framework)
- [Jo.E Framework Paper](https://arxiv.org/abs/...)

## License

MIT License
