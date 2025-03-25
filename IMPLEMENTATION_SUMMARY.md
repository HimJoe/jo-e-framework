# Jo.E Framework Implementation Summary

This repository implements the Joint Evaluation (Jo.E) framework as described in the paper. The implementation consists of the following components:

## 1. Core Framework Architecture

- **`framework/joe_framework.py`**: Main framework class that orchestrates the evaluation pipeline
- **`framework/llm_evaluators/`**: LLM-based initial evaluation components
- **`framework/ai_agents/`**: Specialized AI agents for systematic testing
- **`framework/human_evaluation/`**: Human review interfaces and protocols
- **`framework/scoring/`**: Implementation of the comprehensive scoring system

## 2. Experiments

- **`experiments/experiment1_model_performance/`**: Comparative model performance analysis
- **`experiments/experiment2_domain_specific/`**: Domain-specific evaluation for legal and customer service domains
- **`experiments/experiment3_adversarial/`**: Adversarial robustness testing

## 3. Streamlit App

- **`streamlit_app/app.py`**: Interactive web application for using the Jo.E framework
- **`streamlit_app/README.md`**: Documentation for the Streamlit app
- **`streamlit_app/run_app.sh`**: Startup script for launching the app

## 4. Datasets

- **`datasets/samples/`**: Sample datasets for testing and demonstration

## Key Components Implemented

### LLM Evaluators
- `GPT4Evaluator`: Uses OpenAI's GPT-4o model for initial evaluation
- `Llama3Evaluator`: Uses Meta's Llama 3.2 model for initial evaluation

### AI Agents
- `AdversarialAgent`: Tests for robustness against adversarial inputs
- `BiasDetectionAgent`: Tests for biases across demographic categories
- `KnowledgeVerificationAgent`: Verifies factual accuracy of model outputs
- `EthicalBoundaryAgent`: Tests ethical boundaries and content policies

### Human Review
- `HumanReview`: Interface for expert review of flagged outputs
- Mock review generation for testing and demonstration

### Scoring System
- `ScoringSystem`: Calculates comprehensive scores across dimensions
- `DomainSpecificScoring`: Specialized scoring for specific domains

## Evaluation Process

The Jo.E framework implements a tiered evaluation pipeline:

1. **Initial LLM Screening**: Independent evaluator LLMs classify outputs and flag potential issues
2. **AI Agent Testing**: Specialized agents verify issues through systematic testing
3. **Human Expert Review**: Domain specialists provide final judgments on complex cases
4. **Score Calculation**: Combines results from all evaluation tiers into final scores

## How to Use

### Running Experiments
```bash
cd experiments/experiment1_model_performance
python run_experiment.py
```

### Using the Streamlit App
```bash
cd streamlit_app
streamlit run app.py
```

## Customization

The framework is designed to be extensible:

- Add new LLM evaluators by subclassing `EvaluatorLLM`
- Create specialized agents by subclassing `AgentEvaluator`
- Implement domain-specific scoring by extending `ScoringSystem`
- Add new datasets for specific evaluation scenarios

## Implementation Notes

- The implementation includes mock components for demonstration purposes
- In a production environment, you would replace these with real API calls and integrations
- API keys are required for using commercial LLM services

## Future Improvements

- Add more LLM evaluators (Claude, PaLM, etc.)
- Implement more sophisticated AI agents for specialized domains
- Enhance visualization and reporting capabilities
- Create API endpoints for programmatic access to the framework
- Add more comprehensive test suites and benchmarks
