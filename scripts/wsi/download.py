import os
import openslide
import requests
import json
import random


# Function to query the GDC API for slide files associated with a submitter_id
def query_gdc_api(submitter_id):
    files_endpt = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": [submitter_id]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["SVS"]
                }
            }
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": "100"
    }
    response = requests.get(files_endpt, params=params)
    if response.status_code == 200:
        return response.json()["data"]["hits"]
    else:
        print(f"Error querying GDC API: {response.status_code}")
        return []

# Function to download a file from the GDC API using its file_id
def download_slide(file_id, file_name, download_dir):
    data_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"
    response = requests.get(data_endpt, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(download_dir, file_name)
        with open(file_path, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    output_file.write(chunk)
        print(f"Downloaded {file_name}")
    else:
        print(f"Error downloading {file_name}: {response.status_code}")

# Main function to execute the download process
def download_slides(submitter_id, download_dir, num_slides=10):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Query the GDC API for slide files
    slides = query_gdc_api(submitter_id)

    if not slides:
        print("No slides found for the given submitter_id.")
        return

    # Randomly select the specified number of slides
    selected_slides = random.sample(slides, min(num_slides, len(slides)))

    # Download each selected slide
    for slide in tqdm(selected_slides):
        download_slide(slide["file_id"], slide["file_name"], download_dir)

