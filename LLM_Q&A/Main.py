import os
import json
import torch
import random
from tqdm import tqdm
from ImageCaptioningModel import ImageCaptioningModel
from QAGenerator import QAGenerator

def generate_dataset(image_folder, output_folder):
    """Create dataset from image folder."""
    os.makedirs(output_folder, exist_ok=True)

    caption_model = ImageCaptioningModel()
    qa_generator = QAGenerator(domain="fruit")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_data = []

    for image_file in tqdm(image_files, desc="Sinh dữ liệu"):
        image_path = os.path.join(image_folder, image_file)

        caption = caption_model.generate_caption(image_path)

        qa_pairs = qa_generator.generate_qa_from_caption(caption)

        for qa in qa_pairs:
            data_item = {
                "image_path": image_file,
                "question": qa["question"],
                "answer": qa["answer"],
                "type": qa["type"],
                "caption": caption 
            }
            all_data.append(data_item)
    random.shuffle(all_data)

    n = len(all_data)
    train_data = all_data[:int(0.7 * n)]
    val_data = all_data[int(0.7 * n):int(0.85 * n)]
    test_data = all_data[int(0.85 * n):]

    with open(os.path.join(output_folder, 'train_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_folder, 'val_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_folder, 'test_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Đã tạo xong bộ dữ liệu: {len(train_data)} mẫu train, {len(val_data)} mẫu validation, {len(test_data)} mẫu test")

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

if __name__ == "__main__":
    image_folder = ".\data\images"
    output_folder = ".\data\Json"

    dataset = generate_dataset(image_folder, output_folder)