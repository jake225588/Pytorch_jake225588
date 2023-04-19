import torch
import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import functional as F
from PIL import ImageOps
from PIL import Image
import time


model_file = "runs/train/exp5/weights/best.pt"
yaml_file = "data/coco.yaml"

model = torch.hub.load("ultralytics/yolov5", "custom", path=model_file)

model.eval()


def process_image(model, image_path,result_folder):
    image = cv2.imread(image_path)

    # Convert the image to a PIL image
    pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = F.to_pil_image(pil_image)
    original_width, original_height = pil_image.size
    # Resize the image to a square format while preserving its aspect ratio
    size = 640
    width, height = pil_image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_height = size
        new_width = int(size * aspect_ratio)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    padding = (size - new_width) // 2, (size - new_height) // 2
    pil_image = ImageOps.expand(pil_image, (*padding, size - new_width - padding[0], size - new_height - padding[1]))

    # Run the detection
    with torch.no_grad():
        result = model(pil_image, size=size)

    # Extract the detections
    detections = result.xyxy[0].cpu().numpy()

    # Process the detections
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection

        x1 = (x1 - padding[0]) * original_width / new_width
        x2 = (x2 - padding[0]) * original_width / new_width
        y1 = (y1 - padding[1]) * original_height / new_height
        y2 = (y2 - padding[1]) * original_height / new_height

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Optionally, you can also add a label to the bounding box
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Save the result image

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_extension = os.path.splitext(image_path)[1]
    output_image_name = f"output_{os.path.basename(image_path).replace(file_extension, '')}{file_extension}"
    output_image_path = os.path.join(result_folder, output_image_name)
    cv2.imwrite(output_image_path, image)


    return detections


def run_parallel_detection(model, image_folder, result_folder,max_workers=4):
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(process_image, model, image_path,result_folder) for image_path in image_paths]
        results = [future.result() for future in futures]

    return results

if __name__ == "__main__":
    image_folder = "images/train2017"
    result_folder = 'result'
    max_workers = 8  # Adjust this value based on your available CPU cores
    start_time = time.time()
    results = run_parallel_detection(model, image_folder,result_folder, max_workers)
    end_time = time.time()
    elapsed_time = end_time - start_time
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Processed {len(image_files)} images in {elapsed_time:.2f} seconds")
