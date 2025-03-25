import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys
import logging
import json
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="JO-E Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f6f6f6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .info-text {
        color: #555;
        font-size: 1rem;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 0.5rem;
        border-left: 3px solid #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True)

# Sample datasets for demonstration
def load_sample_data():
    """Load sample datasets for use in the app."""
    # Generate fake attack data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    attacks = {
        'date': dates,
        'attack_type': np.random.choice(['Text Prompt Injection', 'Jailbreak', 'Data Extraction', 'Refusal Bypass', 'Hallucination Inducement'], 100),
        'success_rate': np.random.uniform(0, 1, 100),
        'detection_rate': np.random.uniform(0, 1, 100),
        'model_target': np.random.choice(['GPT-4', 'Claude 3', 'Llama 3', 'Gemini', 'Custom LLM'], 100),
        'mitigation_applied': np.random.choice(['Yes', 'No'], 100)
    }
    return pd.DataFrame(attacks)

# Utility functions
def generate_unique_id():
    """Generate a unique ID for tracking."""
    return str(uuid.uuid4())

def get_timestamp():
    """Get current timestamp in readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_report(report_data, filename="jo-e_assessment_report.json"):
    """Save evaluation report to file."""
    try:
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return False

# Sidebar configuration
def sidebar_config():
    """Configure the sidebar with options and navigation."""
    st.sidebar.title("JO-E Framework")
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=JO-E", width=150)
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Framework Overview", "Evaluation Tools", "Attack Techniques", "Mitigation Strategies", "Case Studies"]
    )
    
    st.sidebar.markdown("---")
    
    # Settings
    with st.sidebar.expander("Settings"):
        theme = st.selectbox("Theme", ["Light", "Dark", "System"])
        language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Japanese"])
        show_advanced = st.checkbox("Show Advanced Options", value=False)
    
    st.sidebar.markdown("---")
    
    # About section
    with st.sidebar.expander("About"):
        st.write("""
        The JO-E Framework is a comprehensive toolkit for evaluating and improving the robustness of LLM systems against adversarial attacks.
        
        Version: 1.0.0
        
        © 2025 JO-E Framework Team
        """)
    
    return page

# Main content components
def home_page():
    """Display home page content."""
    st.markdown('<h1 class="main-header">Welcome to the JO-E Framework</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **J**ailbreak and **O**versight **E**valuation Framework for LLMs
    
    A comprehensive toolkit for evaluating LLM robustness against adversarial attacks.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Evaluate")
        st.write("Test your LLM systems against various attack vectors and identify vulnerabilities.")
        st.button("Start Evaluation", key="eval_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🛡️ Protect")
        st.write("Implement proven defense strategies to enhance your model's robustness.")
        st.button("Explore Defenses", key="defense_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Analyze")
        st.write("Gain insights from attack patterns and defense effectiveness.")
        st.button("View Analytics", key="analytics_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("Framework Overview")
    cols = st.columns(4)
    cols[0].metric("Attack Techniques", "50+")
    cols[1].metric("Defense Strategies", "30+")
    cols[2].metric("Evaluation Metrics", "15+")
    cols[3].metric("Case Studies", "25+")
    
    # Latest updates
    st.markdown("---")
    st.subheader("Latest Updates")
    
    updates = [
        {"date": "2025-03-20", "title": "Added new prompt injection techniques", "category": "Attacks"},
        {"date": "2025-03-15", "title": "Improved detection metrics for jailbreak attempts", "category": "Metrics"},
        {"date": "2025-03-10", "title": "Published case study on healthcare LLM security", "category": "Case Studies"}
    ]
    
    for update in updates:
        st.markdown(f"""
        <div class="highlight">
        <strong>{update['date']}</strong> - {update['title']} <span style="background-color: #e0e0e0; padding: 2px 5px; border-radius: 3px; font-size: 0.8rem;">{update['category']}</span>
        </div>
        """, unsafe_allow_html=True)

def framework_overview():
    """Display framework overview page."""
    st.markdown('<h1 class="main-header">JO-E Framework Overview</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    The JO-E Framework provides a structured approach to evaluating and improving the robustness of LLM systems against adversarial attacks.
    
    Our framework encompasses four key pillars:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1. Threat Modeling")
        st.markdown("""
        - Identify potential attack vectors
        - Assess impact and likelihood
        - Prioritize security concerns
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2. Attack Simulation")
        st.markdown("""
        - Implement known attack techniques
        - Develop novel attack vectors
        - Measure attack success rates
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3. Defense Implementation")
        st.markdown("""
        - Apply proven mitigation strategies
        - Develop custom defenses
        - Measure defense effectiveness
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("4. Continuous Evaluation")
        st.markdown("""
        - Monitor emerging threats
        - Update defense mechanisms
        - Document lessons learned
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Framework diagram
    st.subheader("Framework Diagram")
    # Placeholder for actual diagram
    st.image("https://via.placeholder.com/800x400.png?text=JO-E+Framework+Diagram", use_column_width=True)
    
    st.markdown("---")
    
    # Key benefits
    st.subheader("Key Benefits")
    
    benefits = [
        "**Comprehensive Coverage**: Evaluate against a wide range of attack vectors",
        "**Practical Defenses**: Implement proven strategies to enhance robustness",
        "**Quantitative Metrics**: Measure and track security improvements",
        "**Best Practices**: Follow industry standards for AI safety",
        "**Continuous Improvement**: Stay ahead of emerging threats"
    ]
    
    for benefit in benefits:
        st.markdown(f"- {benefit}")

def evaluation_tools():
    """Display evaluation tools page."""
    st.markdown('<h1 class="main-header">Evaluation Tools</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Our framework provides a suite of tools to evaluate your LLM systems against various attack vectors.
    Select a tool below to get started:
    """)
    
    # Tool selection
    tool = st.selectbox(
        "Select Evaluation Tool",
        ["Prompt Injection Tester", "Jailbreak Detector", "Data Extraction Scanner", "Policy Violation Checker", "Custom Attack Simulator"]
    )
    
    st.markdown("---")
    
    if tool == "Prompt Injection Tester":
        st.subheader("Prompt Injection Tester")
        
        st.markdown("""
        Test your LLM's resilience against attempts to override system instructions through carefully crafted prompts.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Configuration")
            model_name = st.text_input("Model Name/API", "gpt-4-turbo")
            system_prompt = st.text_area("System Prompt", "You are a helpful assistant. Do not provide harmful information.")
            test_cases = st.number_input("Number of Test Cases", min_value=1, max_value=100, value=10)
            
            st.markdown("#### Advanced Settings")
            with st.expander("Advanced Options"):
                attack_strength = st.slider("Attack Strength", 1, 10, 5)
                use_templates = st.checkbox("Use Attack Templates", value=True)
                custom_templates = st.text_area("Custom Attack Templates (one per line)", "")
        
        with col2:
            st.markdown("#### Run Test")
            st.warning("This will submit multiple prompts to your LLM. API costs may apply.")
            
            if st.button("Start Testing", key="start_injection_test"):
                # Placeholder for actual testing
                with st.spinner("Running tests..."):
                    # Simulate processing time
                    import time
                    time.sleep(2)
                
                st.success("Testing completed!")
                
                # Sample results
                results = {
                    "total_tests": 10,
                    "successful_injections": 3,
                    "blocked_attempts": 7,
                    "success_rate": "30%",
                    "vulnerabilities_found": [
                        "System prompt override via unicode obfuscation",
                        "Instruction injection via markdown formatting",
                        "Context manipulation through token stuffing"
                    ]
                }
                
                st.markdown("#### Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Tests", results["total_tests"])
                col2.metric("Successful Injections", results["successful_injections"])
                col3.metric("Blocked Attempts", results["blocked_attempts"])
                col4.metric("Success Rate", results["success_rate"])
                
                st.markdown("#### Vulnerabilities Found")
                for vuln in results["vulnerabilities_found"]:
                    st.markdown(f"- {vuln}")
                
                st.markdown("#### Recommendations")
                st.markdown("""
                - Strengthen your system prompt with explicit constraints
                - Implement input sanitization for special characters
                - Add detection mechanisms for common injection patterns
                """)
    
    elif tool == "Jailbreak Detector":
        st.subheader("Jailbreak Detector")
        # Similar structure to the Prompt Injection Tester
        st.markdown("Tool under development. Coming soon!")
    
    else:
        st.info(f"The {tool} is currently under development. Please check back soon!")

def attack_techniques():
    """Display attack techniques page."""
    st.markdown('<h1 class="main-header">Attack Techniques Library</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Browse our comprehensive library of LLM attack techniques, categorized by type, impact, and complexity.
    """)
    
    # Attack categories
    attack_categories = {
        "Prompt Injection": "Attacks that manipulate the model's instructions",
        "Jailbreaking": "Techniques to bypass content policy constraints",
        "Data Extraction": "Methods to extract sensitive information or training data",
        "Refusal Bypassing": "Techniques to overcome model refusal on harmful content",
        "Hallucination Inducement": "Methods to force the model to generate false information"
    }
    
    # Category selection
    category = st.selectbox("Select Attack Category", list(attack_categories.keys()))
    
    st.markdown(f"**{category}**: {attack_categories[category]}")
    
    st.markdown("---")
    
    # Sample data
    sample_data = load_sample_data()
    category_data = sample_data[sample_data['attack_type'] == category]
    
    # Statistics
    st.subheader("Category Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Success Rate", f"{category_data['success_rate'].mean():.1%}")
        st.metric("Detection Rate", f"{category_data['detection_rate'].mean():.1%}")
    
    with col2:
        # Create a pie chart of model targets
        fig, ax = plt.subplots(figsize=(5, 5))
        category_data['model_target'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Model Targets')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Techniques list
    st.subheader("Techniques")
    
    techniques = [
        {"name": f"{category} Technique 1", "description": "Description of the technique", "complexity": "Low", "impact": "Medium"},
        {"name": f"{category} Technique 2", "description": "Description of the technique", "complexity": "Medium", "impact": "High"},
        {"name": f"{category} Technique 3", "description": "Description of the technique", "complexity": "High", "impact": "Critical"}
    ]
    
    for i, technique in enumerate(techniques):
        with st.expander(f"{i+1}. {technique['name']} (Complexity: {technique['complexity']}, Impact: {technique['impact']})"):
            st.markdown(f"**Description**: {technique['description']}")
            st.markdown("**Example**:")
            st.code(f"Example of {technique['name']}")
            st.markdown("**Mitigation**:")
            st.markdown(f"- Mitigation strategy 1 for {technique['name']}")
            st.markdown(f"- Mitigation strategy 2 for {technique['name']}")

def mitigation_strategies():
    """Display mitigation strategies page."""
    st.markdown('<h1 class="main-header">Mitigation Strategies</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore effective mitigation strategies to defend your LLM systems against various attack vectors.
    """)
    
    # Defense categories
    defense_categories = {
        "Prompt Engineering": "Techniques to craft robust system prompts",
        "Input Validation": "Methods to validate and sanitize user inputs",
        "Output Filtering": "Approaches to filter potentially harmful outputs",
        "Monitoring & Detection": "Systems to monitor and detect attack attempts",
        "Architecture Modifications": "Changes to model architecture for improved security"
    }
    
    # Category selection
    category = st.selectbox("Select Defense Category", list(defense_categories.keys()))
    
    st.markdown(f"**{category}**: {defense_categories[category]}")
    
    st.markdown("---")
    
    # Effectiveness chart
    st.subheader("Effectiveness Against Attack Types")
    
    # Sample data for chart
    attack_types = ["Prompt Injection", "Jailbreak", "Data Extraction", "Refusal Bypass", "Hallucination"]
    effectiveness = np.random.uniform(0.5, 0.95, len(attack_types))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(attack_types, effectiveness)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Effectiveness')
    ax.set_title(f'{category} Effectiveness Against Attack Types')
    
    # Add labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{effectiveness[i]:.0%}',
                va='center')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Strategies list
    st.subheader("Strategies")
    
    strategies = [
        {"name": f"{category} Strategy 1", "description": "Description of the strategy", "complexity": "Low", "effectiveness": "Medium"},
        {"name": f"{category} Strategy 2", "description": "Description of the strategy", "complexity": "Medium", "effectiveness": "High"},
        {"name": f"{category} Strategy 3", "description": "Description of the strategy", "complexity": "High", "effectiveness": "Very High"}
    ]
    
    for i, strategy in enumerate(strategies):
        with st.expander(f"{i+1}. {strategy['name']} (Complexity: {strategy['complexity']}, Effectiveness: {strategy['effectiveness']})"):
            st.markdown(f"**Description**: {strategy['description']}")
            st.markdown("**Implementation**:")
            st.code(f"Implementation example for {strategy['name']}")
            st.markdown("**Trade-offs**:")
            st.markdown(f"- Trade-off 1 for {strategy['name']}")
            st.markdown(f"- Trade-off 2 for {strategy['name']}")
            st.markdown("**When to use**:")
            st.markdown(f"Guidance on when to use {strategy['name']}")

def case_studies():
    """Display case studies page."""
    st.markdown('<h1 class="main-header">Case Studies</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Real-world examples of LLM security assessments, attacks, and mitigations.
    """)
    
    # Case study categories
    categories = ["Healthcare", "Finance", "Education", "Government", "Enterprise", "Consumer"]
    category = st.selectbox("Filter by Sector", ["All"] + categories)
    
    st.markdown("---")
    
    # List of case studies
    case_studies = [
        {
            "title": "Healthcare Chatbot Jailbreak Analysis",
            "sector": "Healthcare",
            "date": "2025-02-15",
            "summary": "Analysis of jailbreak attempts on a medical advice chatbot",
            "key_findings": ["Vulnerability to indirect prompt injection", "Successful extraction of training data patterns", "Effective mitigation via input preprocessing"]
        },
        {
            "title": "Financial Advisor LLM Security Assessment",
            "sector": "Finance",
            "date": "2025-01-20",
            "summary": "Comprehensive security assessment of an LLM used for financial advice",
            "key_findings": ["Robust against direct jailbreak attempts", "Vulnerable to multi-turn social engineering", "Improved with enhanced monitoring"]
        },
        {
            "title": "Education Platform Data Leakage Study",
            "sector": "Education",
            "date": "2024-12-10",
            "summary": "Investigation of potential training data leakage in an educational AI tutor",
            "key_findings": ["Memorization of copyrighted educational content", "Inconsistent PII handling", "Successful mitigation via fine-tuning"]
        }
    ]
    
    # Filter by category if needed
    if category != "All":
        case_studies = [cs for cs in case_studies if cs["sector"] == category]
    
    # Display case studies
    for case_study in case_studies:
        with st.expander(f"{case_study['title']} ({case_study['sector']}, {case_study['date']})"):
            st.markdown(f"**Summary**: {case_study['summary']}")
            st.markdown("**Key Findings**:")
            for finding in case_study["key_findings"]:
                st.markdown(f"- {finding}")
            st.markdown("**Learn More**:")
            st.markdown("[Read Full Case Study](#)")  # Placeholder link

def display_main_interface():
    """Display the main interface based on the selected page."""
    # Apply custom CSS
    load_css()
    
    # Get the selected page from sidebar
    page = sidebar_config()
    
    # Display the appropriate page
    if page == "Home":
        home_page()
    elif page == "Framework Overview":
        framework_overview()
    elif page == "Evaluation Tools":
        evaluation_tools()
    elif page == "Attack Techniques":
        attack_techniques()
    elif page == "Mitigation Strategies":
        mitigation_strategies()
    elif page == "Case Studies":
        case_studies()

def main():
    """Main application entry point."""
    
    # Configure sidebar
    sidebar_config()
    
    # Display main interface
    display_main_interface()

if __name__ == "__main__":
    main()
