"""
Experiment 1: Comparative Model Performance Analysis

This script runs Experiment 1 from the Jo.E paper, comparing
the performance of multiple foundation models using the Jo.E evaluation framework.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path so we can import from framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework.joe_framework import JoEFramework
from framework.llm_evaluators.evaluator_llm import GPT4Evaluator, Llama3Evaluator
from framework.ai_agents.agent_evaluator import (
    AdversarialAgent, BiasDetectionAgent, 
    KnowledgeVerificationAgent, EthicalBoundaryAgent
)
from framework.human_evaluation.human_review import HumanReview
from framework.scoring.scoring_system import ScoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment1.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_datasets(base_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load evaluation datasets.
    
    Args:
        base_path: Base path to dataset directory
        
    Returns:
        Dictionary mapping dataset names to lists of examples
    """
    datasets = {}
    
    dataset_files = {
        "general_knowledge": "general_knowledge.json",
        "reasoning": "reasoning.json",
        "creative": "creative.json",
        "sensitive": "sensitive.json"
    }
    
    for dataset_name, filename in dataset_files.items():
        try:
            path = os.path.join(base_path, filename)
            with open(path, 'r') as f:
                data = json.load(f)
            datasets[dataset_name] = data
            logger.info(f"Loaded {len(data)} examples from {dataset_name} dataset")
        except FileNotFoundError:
            logger.warning(f"Dataset file not found: {path}")
            # Create a small placeholder dataset for testing
            datasets[dataset_name] = [
                {"id": f"{dataset_name}_1", "prompt": f"Example prompt for {dataset_name}"},
                {"id": f"{dataset_name}_2", "prompt": f"Another example for {dataset_name}"}
            ]
    
    return datasets

def get_model_responses(
    model_name: str, 
    datasets: Dict[str, List[Dict[str, Any]]],
    api_key: str = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get model responses for the datasets.
    
    Args:
        model_name: Name of the model to use
        datasets: Datasets to get responses for
        api_key: API key for model access
        
    Returns:
        Dictionary mapping dataset names to lists of model outputs
    """
    logger.info(f"Getting responses from model: {model_name}")
    
    # In a real implementation, this would call the appropriate model API
    # Here we're simulating the model responses
    
    all_responses = {}
    for dataset_name, examples in datasets.items():
        responses = []
        for example in examples:
            prompt = example["prompt"]
            example_id = example["id"]
            
            # Simulate model response
            fake_response = f"This is a simulated response from {model_name} for the prompt: {prompt[:50]}..."
            
            responses.append({
                "id": example_id,
                "prompt": prompt,
                "response": fake_response
            })
        
        all_responses[dataset_name] = responses
        logger.info(f"Generated {len(responses)} responses for {dataset_name} dataset with {model_name}")
    
    return all_responses

def initialize_framework(args) -> JoEFramework:
    """
    Initialize the Jo.E framework with components.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Initialized JoEFramework instance
    """
    # Initialize LLM evaluators
    evaluator_llms = []
    
    if args.use_gpt4:
        try:
            gpt4_evaluator = GPT4Evaluator(api_key=args.openai_api_key)
            evaluator_llms.append(gpt4_evaluator)
            logger.info("Initialized GPT-4o evaluator")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4o evaluator: {e}")
    
    if args.use_llama:
        try:
            llama_evaluator = Llama3Evaluator(
                api_key=args.llama_api_key,
                model_path=args.llama_model_path
            )
            evaluator_llms.append(llama_evaluator)
            logger.info("Initialized Llama-3.2 evaluator")
        except Exception as e:
            logger.error(f"Failed to initialize Llama-3.2 evaluator: {e}")
    
    # Initialize AI agents
    ai_agents = [
        AdversarialAgent(name="adversarial_agent"),
        BiasDetectionAgent(name="bias_detection_agent"),
        KnowledgeVerificationAgent(name="knowledge_verification_agent"),
        EthicalBoundaryAgent(name="ethical_boundary_agent")
    ]
    logger.info(f"Initialized {len(ai_agents)} AI agents")
    
    # Initialize human review component
    human_review = HumanReview(
        mock_reviews=True,  # Use mock reviews for testing
        review_storage_path=args.review_storage_path
    )
    logger.info("Initialized human review component with mock reviews")
    
    # Initialize the framework
    framework = JoEFramework(
        evaluator_llms=evaluator_llms,
        ai_agents=ai_agents,
        human_review=human_review,
        enable_human_review=args.enable_human_review
    )
    logger.info("Jo.E framework initialized successfully")
    
    return framework

def run_experiment(args):
    """
    Run the model performance comparison experiment.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting Experiment 1: Comparative Model Performance Analysis")
    
    # Load datasets
    datasets = load_datasets(args.dataset_path)
    
    # Initialize the framework
    framework = initialize_framework(args)
    
    # Run evaluation for each model
    models = ["GPT-4o", "Llama-3.2", "Phi-3"]
    results = {}
    
    for model in models:
        if model == "GPT-4o" and not args.evaluate_gpt4:
            logger.info(f"Skipping evaluation of {model}")
            continue
            
        if model == "Llama-3.2" and not args.evaluate_llama:
            logger.info(f"Skipping evaluation of {model}")
            continue
            
        if model == "Phi-3" and not args.evaluate_phi:
            logger.info(f"Skipping evaluation of {model}")
            continue
        
        logger.info(f"Evaluating model: {model}")
        
        # Get model responses
        model_responses = get_model_responses(model, datasets, api_key=args.openai_api_key)
        
        # Combine all responses for evaluation
        all_outputs = []
        for dataset_responses in model_responses.values():
            all_outputs.extend(dataset_responses)
        
        # Run the Jo.E evaluation pipeline
        evaluation_results = framework.evaluate(all_outputs, model)
        
        # Store the results
        results[model] = evaluation_results
        
        logger.info(f"Completed evaluation of {model}. Jo.E score: {evaluation_results['scores']['joe_score']}")
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"experiment1_results_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment results saved to {output_path}")
    
    # Print summary
    logger.info("Experiment 1 Summary:")
    for model, evaluation in results.items():
        scores = evaluation["scores"]
        logger.info(f"Model: {model}")
        logger.info(f"  Jo.E Score: {scores['joe_score']}")
        logger.info(f"  Accuracy: {scores['accuracy_score']}")
        logger.info(f"  Robustness: {scores['robustness_score']}")
        logger.info(f"  Fairness: {scores['fairness_score']}")
        logger.info(f"  Ethics: {scores['ethics_score']}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Experiment 1: Comparative Model Performance Analysis")
    
    parser.add_argument("--dataset_path", type=str, default="../datasets",
                        help="Path to the dataset directory")
    parser.add_argument("--output_path", type=str, default="./results",
                        help="Path to save experiment results")
    parser.add_argument("--review_storage_path", type=str, default="./reviews",
                        help="Path to store human reviews")
    
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--llama_api_key", type=str, default=None,
                        help="Llama API key")
    parser.add_argument("--llama_model_path", type=str, default=None,
                        help="Path to local Llama model weights")
    
    parser.add_argument("--use_gpt4", action="store_true",
                        help="Use GPT-4o for evaluation")
    parser.add_argument("--use_llama", action="store_true",
                        help="Use Llama-3.2 for evaluation")
    
    parser.add_argument("--evaluate_gpt4", action="store_true",
                        help="Evaluate GPT-4o model")
    parser.add_argument("--evaluate_llama", action="store_true",
                        help="Evaluate Llama-3.2 model")
    parser.add_argument("--evaluate_phi", action="store_true",
                        help="Evaluate Phi-3 model")
    
    parser.add_argument("--enable_human_review", action="store_true",
                        help="Enable human review component")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.review_storage_path, exist_ok=True)
    
    try:
        run_experiment(args)
    except Exception as e:
        logger.exception(f"Experiment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
