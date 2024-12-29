from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import os
import shutil
import base64
import requests
from PIL import Image
from dotenv import load_dotenv

# LOAD API KEY FROM .ENV FILE
load_dotenv()
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize YOLO model
model = YOLO('weights.pt')

# OpenLibrary API headers
headers = {
    "User-Agent": "PostmanRuntime/7.29.0",
    "Accept": "*/*",
    "Cache-Control": "no-cache",
    "Host": "openlibrary.org",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Helper function to rotate images
def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Helper function to encode images to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# API endpoint
@app.post("/process-bookshelf/")
async def process_bookshelf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        input_image_path = f"input_{file.filename}"
        with open(input_image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # YOLO detection
        results = model.predict(source=input_image_path, conf=0.5, save=False)
        bounding_boxes = results[0].boxes.xyxy.numpy()

        # Create directory for cropped images
        os.makedirs("cropped_spines", exist_ok=True)
        cropped_paths = []

        # Crop and rotate images
        image = cv2.imread(input_image_path)
        for i, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            cropped = image[y_min:y_max, x_min:x_max]
            rotated_cropped = rotate_image(cropped)
            cropped_path = f"cropped_spines/spine_{i}.jpg"
            cv2.imwrite(cropped_path, rotated_cropped)
            cropped_paths.append(cropped_path)

        # Extract book titles and authors using GPT Vision API
        gpt_api_key = OPEN_AI_KEY

        from openai import OpenAI
        client = OpenAI(api_key=gpt_api_key)

        results = []
        for cropped_path in cropped_paths:
            base64_image = encode_image(cropped_path)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the book title and author name from this image. Format the response as '{book title} {author name}'."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=300,
            )
            detected_text = response.choices[0].message.content.strip()
            results.append({"image": cropped_path, "text": detected_text})

        # Query OpenLibrary API for book details
        book_objects = []
        base_url = "https://openlibrary.org/search.json"
        for result in results:
            title = result["text"]
            response = requests.get(base_url, params={"q": title}, headers=headers, timeout=15)
            # print(response.json())
            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])
                if docs:
                    book_info = docs[0]
                    # print(book_info)
                    book_objects.append({
                        "title": book_info.get("title", "N/A"),
                        "author_name": book_info.get("author_name", ["N/A"])[0],
                        "first_publish_year": book_info.get("first_publish_year", "N/A"),
                        "number_of_pages_median": book_info.get("number_of_pages_median", "N/A"),
                    })
                    print({
                        "title": book_info.get("title", "N/A"),
                        "author_name": book_info.get("author_name", ["N/A"])[0],
                        "first_publish_year": book_info.get("first_publish_year", "N/A"),
                        "number_of_pages_median": book_info.get("number_of_pages_median", "N/A"),
                    })
                else:
                    book_objects.append({"title": "N/A"})
            else:
                book_objects.append({"title": "Error querying OpenLibrary API"})

        return {"books": book_objects}

    except Exception as e:
        return {"error": str(e)}

