import requests
import uuid
import time
import base64
import json

# Use the image path provided by the system after upload
image_file = 'C:/Users/Hoon/Downloads/others/-500/img/displacement_294s.png'
output_json_file = 'C:/Users/Hoon/Downloads/others/-500/img/result.json'

api_url  = 'api_url'

secret_key = 'secret_key'

def getEquation():
    with open(image_file, 'rb') as f:
        file_data = f.read()

    request_json = {
        'images': [
            {
                'format': 'png',
                'name': 'demo',
                'data': base64.b64encode(file_data).decode()
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = json.dumps(request_json).encode('UTF-8')
    headers = {
        'X-OCR-SECRET': secret_key,
        'Content-Type': 'application/json'
    }

    session = requests.Session()
    try:
        response = session.post(api_url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        extracted_texts = [field['inferText'] for field in json.loads(response.text)["images"][0]['fields']]
        return extracted_texts
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# Extracted texts and save them as JSON
texts = getEquation()
save_to_json(texts, output_json_file)

print(f"Extracted text has been saved to {output_json_file}.")
