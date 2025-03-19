import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, VisionEncoderDecoderModel, ViTImageProcessor
from PIL import Image

class ImageCaptioningModel:
    def __init__(self):
        # Use ViT-GPT2 to create captions for images
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate caption
        output_ids = self.model.generate(
            pixel_values,
            max_length=30,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return caption
