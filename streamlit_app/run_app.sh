#!/bin/bash

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit
fi

# Create reviews directory if it doesn't exist
mkdir -p ./reviews

# Run the Streamlit app
echo "Starting Jo.E Framework App..."
streamlit run app.py
