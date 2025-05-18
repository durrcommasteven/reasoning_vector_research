#!/bin/bash

apt update 
apt install -y vim screen

REQUIRED_PYTHON="3.10"

version_ge() {
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1" ]
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not be found."
        return 1
    fi

    # Get the current Python version as a string
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

    # Check if the current Python version is greater than or equal to the required version
    if ! version_ge "$PYTHON_VERSION" "$REQUIRED_PYTHON"; then
        echo "This script requires Python $REQUIRED_PYTHON or higher but found $PYTHON_VERSION"
        return 1
    fi

    return 0
}

setup_git() {
    echo "Are you Steven Durr? (it is a federal crime punishable by death to lie to me) (y/n)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        git config --global user.name "Steven Durr"
        git config --global user.email "durrcommasteven@gmail.com"
    else
        echo "Please enter your Git user.name:"
        read -r name
        echo "Please enter your Git user.email:"
        read -r email
        git config --global user.name "$name"
        git config --global user.email "$email"
    fi
}

setup_hf() {
    echo "Please enter your Hugging Face token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Storing HF_TOKEN in .env file..."
        echo "HF_TOKEN=$token" >> .env
        
        echo "Installing Hugging Face CLI..."
        yes | pip install --upgrade huggingface_hub
        echo "Logging in to Hugging Face CLI..."
        huggingface-cli login --token $token
    else
        echo "No token entered. Skipping..."
    fi
}

setup_together() {
    echo "Please enter your Together AI token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Storing TOGETHER_API_KEY in .env file..."
        echo "TOGETHER_API_KEY=$token" >> .env
    else
        echo "No token entered. Skipping..."
    fi
}

setup_venv() {
    echo "Setting up venv..."

    python -m venv venv
    source venv/bin/activate

    echo "Done setting up venv!"
}

install_requirements() {
    echo "Installing requirements..."

    yes | pip install -r requirements.txt --upgrade

    echo "Done installing requirements!"
}

download_resid_data() {
    echo "Downloading reasoning_resid_data from GCS…"

    export GOOGLE_APPLICATION_CREDENTIALS="/root/key.json"

    if [ -z "$GCS_BUCKET" ]; then
        echo "❌ ERROR: GCS_BUCKET environment variable not set."
        return 1
    fi

    pip install --upgrade google-cloud-storage

    python3 - <<'EOF'
import os
from google.cloud import storage

bucket_name = os.environ["GCS_BUCKET"]
client = storage.Client()
bucket = client.bucket(bucket_name)

for blob in client.list_blobs(bucket, prefix="reasoning_resid_data/"):
    if blob.name.endswith("/"): continue
    os.makedirs(os.path.dirname(blob.name), exist_ok=True)
    blob.download_to_filename(blob.name)

print("✅ reasoning_resid_data downloaded")
EOF
}

setup_gcs_key() {
    echo ""
    echo "🔐 Paste your GCS service account JSON below, then press Ctrl-D (or Ctrl-Z then Enter):"
    mkdir -p /root
    cat > /root/key.json
    echo ""
    echo "✅ Key saved to /root/key.json"
}

echo "Running set up..."

echo "" > .env

check_python
if [ $? -ne 0 ]; then
    return 1
fi

setup_git
setup_hf
setup_venv
install_requirements
setup_gcs_key
download_resid_data

echo "All set up!"
