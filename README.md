# Social Media Opinions Analysis with LLMs

This project analyzes social media opinions using Large Language Models (LLMs) by grouping, classifying, and summarizing them.

## Getting Started

### Prerequisites

- Git
- Git LFS
- Conda (Miniconda or Anaconda)

### Cloning the Repository

1. Install Git LFS if you haven't already:
   ```
   git lfs install
   ```

2. Clone the repository:
   ```
   git clone https://github.com/MoaazK/social-media-opinions-analysis-with-llms.git
   cd social-media-opinions-analysis-with-llms
   ```

3. Pull the LFS files:
   ```
   git lfs pull
   ```

   Note: There's no need to run `git pull` after `git lfs pull` as the regular files are already up to date from the initial clone.

### Setting up the Environment

1. Create a new Conda environment with Python 3.12:
   ```
   conda create -n social-media-analysis python=3.12
   ```

2. Activate the environment:
   ```
   conda activate social-media-analysis
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Using the Trained Models

This project includes pre-trained models for:
- Comments grouping
- Classification
- Summarization

These models are located in the `saved_models` folder.

## Usage

### Running the Sample Script

To run the sample analysis script:

1. Make sure you're in the project directory and your Conda environment is activated.

2. Run the main script:
   ```
   python main.py
   ```

This script demonstrates how to use the trained models for analyzing social media opinions.

### Using the API Service

To use the models through an API:

1. Make sure you're in the project directory and your Conda environment is activated.

2. Start the FastAPI server:
   ```
   uvicorn api:app --reload
   ```

3. The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

4. Use the API endpoints to analyze social media opinions programmatically.

#### Example: Using Python requests

Here's an example of how to send a POST request to the `/process_opinions` endpoint using Python:

```python
import requests
import json

# API endpoint URL
url = "http://localhost:8000/process_opinions"

# Sample list of texts to analyze
texts = [
    "I believe that the face on Mars is a natural formation.",
    "I think the FACS has a good chance of changing the future in a possitive way.",
    "They could be happy, sad, surprise, angery, disgusted,and afraid.",
    "Mars has many mysteries that we have yet to uncover.",
    "I think space exploration is a waste of resources.",
    "Joining an extracurricular activity is a great way to make new friends, because you get to meet people that you would of never talked to if you didn't join that club."
]

# Prepare the request payload
payload = {
    "texts": texts
}

# Set the headers
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    print("API Response:")
    print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

#### Using cURL

You can use cURL to send a POST request to the API:

```bash
curl -X 'POST' \
  'http://localhost:8000/process_opinions' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
        "I believe that the face on Mars is a natural formation.",
        "I think the FACS has a good chance of changing the future in a possitive way.",
        "They could be happy, sad, surprise, angery, disgusted,and afraid.",
        "Mars has many mysteries that we have yet to uncover.",
        "I think space exploration is a waste of resources.",
        "Joining an extracurricular activity is a great way to make new friends, because you get to meet people that you would of never talked to if you didn't join that club."
    ]
}'
```

#### Using Postman

1. Open Postman and create a new request.
2. Set the request type to POST.
3. Enter the URL: `http://localhost:8000/process_opinions`
4. Go to the "Body" tab, select "raw", and choose "JSON" from the dropdown.
5. Enter the following JSON in the request body:
   ```json
   {
        "texts": [
            "I believe that the face on Mars is a natural formation.",
            "I think the FACS has a good chance of changing the future in a possitive way.",
            "They could be happy, sad, surprise, angery, disgusted,and afraid.",
            "Mars has many mysteries that we have yet to uncover.",
            "I think space exploration is a waste of resources.",
            "Joining an extracurricular activity is a great way to make new friends, because you get to meet people that you would of never talked to if you didn't join that club."
        ]
   }
   ```
6. Click "Send" to make the request.

#### Using Swagger UI

1. Open your web browser and go to `http://localhost:8000/docs`
2. Find the `/process_opinions` endpoint and click on it to expand.
3. Click the "Try it out" button.
4. In the Request body field, enter the JSON payload:
   ```json
   {
     "texts": [
        "I believe that the face on Mars is a natural formation.",
        "I think the FACS has a good chance of changing the future in a possitive way.",
        "They could be happy, sad, surprise, angery, disgusted,and afraid.",
        "Mars has many mysteries that we have yet to uncover.",
        "I think space exploration is a waste of resources.",
        "Joining an extracurricular activity is a great way to make new friends, because you get to meet people that you would of never talked to if you didn't join that club."
    ]
   }
   ```
5. Click the "Execute" button to send the request.
6. The response will be displayed in the Responses section below.
