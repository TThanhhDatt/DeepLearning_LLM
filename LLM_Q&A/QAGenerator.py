import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

class QAGenerator:
    def __init__(self, domain="fruit"):
        # GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set up fields and question types
        self.domain = domain
        self.question_types = ["count", "identify"]

        # Create question templates for each type
        self.question_templates = {
            "count": [
                "How many {object} are in the picture?",
                "Count the number of {object} in the image.",
                "How many {object} do you see in this picture??",
                "How many {object} are in the image?",
                "Count how many {object} are present in the image."
            ],
            "identify": [
                "What kind of fruit is in the picture?",
                "What fruits do you see in the picture?",
                "Identify the fruits in the picture.",
                "What fruit appears in this picture?",
                "What fruits do you see in the picture?"
            ]
        }

        # Create answer templates
        self.answer_templates = {
            "count": [
                "Have {number} {object}.",
                "There are {number} {object} in the picture.",
                "I see {number} {object} in the picture.",
                "There are a total of {number} {object} in the image."
            ],
            "identify": [
                "The picture contains {fruits}.",
                "I see {fruits} in the picture.",
                "The fruits in the picture include {fruits}.",
                "Image contains {fruits}."
            ]
        }

        self.fruit_names = [
            "apple", "banana", "orange", "strawberry", "kiwi",
            "grapes", "mango", "chickoo", "cherry"
        ]

    def generate_qa_from_caption(self, caption):
        """Generate questions and answers from image captions."""
        # Analyze captions to identify fruits and quantities
        fruits_in_caption = []
        for fruit in self.fruit_names:
            if fruit in caption.lower():
                fruits_in_caption.append(fruit)

        if not fruits_in_caption:
            fruits_in_caption = ["fruits"]

        qa_pairs = []

        # Create questions and answers
        for q_type in self.question_types:
          if q_type == "count":
            for fruit in fruits_in_caption:
              number = random.randint(1, 8)
              question_template = random.choice(self.question_templates[q_type])
              answer_template = random.choice(self.answer_templates[q_type])

              question = question_template.format(object=fruit)
              answer = answer_template.format(number=number, object=fruit)

              qa_pairs.append({
                  "question": question,
                  "answer": answer,
                  "type": q_type
              })

          elif q_type == "identify" and len(fruits_in_caption) > 0:
            question = random.choice(self.question_templates[q_type])

            if len(fruits_in_caption) == 1:
              fruits_text = fruits_in_caption[0]
            else:
              if len(fruits_in_caption) == 2:
                fruits_text = f"{fruits_in_caption[0]} and {fruits_in_caption[1]}"
              else:
                fruits_text = ", ".join(fruits_in_caption[:-1]) + f" and {fruits_in_caption[-1]}"

              answer = random.choice(self.answer_templates[q_type]).format(fruits=fruits_text)

              qa_pairs.append({
                  "question": question,
                  "answer": answer,
                  "type": q_type
              })

        # Using GPT-2 to refine answers
        refined_qa_pairs = self.refine_with_gpt2(qa_pairs, caption)

        return refined_qa_pairs

    def refine_with_gpt2(self, qa_pairs, caption):
        """Use GPT-2 to refine the answer."""
        refined_pairs = []

        for qa in qa_pairs:
            prompt = f"Based on photo description: '{caption}'. \nQuestion: {qa['question']}\nAnswer:"

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(self.device)

            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 30,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer_start = generated_text.find("Correct and complete answer:") + len("Correct and complete answer:")
            refined_answer = generated_text[answer_start:].strip()
            if not refined_answer or len(refined_answer) < 3:
                refined_answer = qa['answer']

            refined_pairs.append({
                "question": qa['question'],
                "answer": refined_answer,
                "type": qa['type']
            })

        return refined_pairs