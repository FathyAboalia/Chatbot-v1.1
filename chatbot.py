# chatbot.py

import re
import os
import logging
from typing import Tuple, Optional
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        try:
            # Initialize Flan-T5 model and tokenizer with PyTorch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info(f"Flan-T5 model ({model_name}) initialized successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to initialize Flan-T5 model: {str(e)}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using Flan-T5 for a given prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Error generating text with Flan-T5: {str(e)}")
            raise

    def process_input(self, user_input: str) -> Tuple[str, Optional[str], int]:
        """Process user input to extract intent, product, and quantity using Flan-T5."""
        user_input_lower = user_input  # Keep original case for product names

        # Define possible intents
        INTENTS = ["place_order"]

        # Try keyword-based intent detection first for efficiency
        if any(kw in user_input_lower.lower() for kw in ["order", "buy", "place"]):
            intent = "place_order"
        else:
            # Use Flan-T5 for intent classification
            intent_prompt = (
                f"Classify the intent of the following input as one of {', '.join(INTENTS)}: "
                f"'{user_input}'"
            )
            intent = self.generate_text(intent_prompt)
            if intent not in INTENTS:
                intent = "unknown"  # Fallback for unrecognized intents

        # Use Flan-T5 to extract product and quantity
        extraction_prompt = (
            f"Extract the product name and quantity from the following input. "
            f"Return in the format: 'product: <name>, quantity: <number>'. "
            f"If no product or quantity is found, use 'product: None, quantity: 1'. "
            f"Input: '{user_input}'"
        )
        extraction_result = self.generate_text(extraction_prompt)

        # Parse the extraction result
        try:
            product_match = re.search(r"product: (.*?)(?:, quantity:|$)", extraction_result)
            quantity_match = re.search(r"quantity: (\d+)", extraction_result)
            product = product_match.group(1).strip() if product_match and product_match.group(1).strip() != "None" else None
            quantity = int(quantity_match.group(1)) if quantity_match else 1
        except Exception as e:
            logger.warning(f"Failed to parse extraction result: {extraction_result}. Error: {str(e)}")
            product = None
            quantity = 1

        logger.info(f"Extracted - Intent: {intent}, Product: {product}, Quantity: {quantity}")

        return intent, product, quantity

class Chatbot:
    def __init__(self, service_layer, nlp_processor: NLPProcessor):
        self.db = service_layer
        self.nlp = nlp_processor

    def get_response(self, user_input: str) -> str:
        try:
            intent, product, quantity = self.nlp.process_input(user_input)

            # Detect language preference based on input
            is_arabic = any(char in user_input for char in "ابتثجحخدذرزسشصضطظعغفقكلمنهويةى")

            if not product:
                return "من فضلك حدد اسم المنتج." if is_arabic else "Please specify the product name."

            if intent == "place_order":
                result = self.db.place_order(product, quantity)
                return f"تم تسجيل طلبك لـ {quantity} وحدة من {product}." if is_arabic else result

            elif intent == "check_stock":
                stock = self.db.check_stock(product)
                return f"مخزون {product} حالياً هو {stock} وحدة." if is_arabic else f"Current stock of {product} is {stock} units."

            # Use Flan-T5 for generating a fallback response
            fallback_prompt = (
                f"Generate a polite response in {'Arabic' if is_arabic else 'English'} for a chatbot "
                f"when the user's request is unclear. Input: '{user_input}'"
            )
            return self.nlp.generate_text(fallback_prompt)
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return "حدث خطأ، من فضلك حاول مرة أخرى." if is_arabic else "An error occurred. Please try again."

def create_chatbot():
    try:
        from sapb1 import SAPB1ServiceLayer  # Import here to avoid circular import
        service_layer = SAPB1ServiceLayer(
            base_url=os.getenv("SAPB1_BASE_URL", "https://c19807sl01d04.cloudiax.com:50000/b1s/v1"),
            company=os.getenv("SAPB1_COMPANY", "A19807_DEMO"),
            username=os.getenv("SAPB1_USERNAME", "manager"),
            password=os.getenv("SAPB1_PASSWORD", "12345"),
            verify_ssl=os.getenv("SAPB1_VERIFY_SSL", "True").lower() == "true"
        )
        nlp = NLPProcessor()
        return Chatbot(service_layer, nlp)
    except Exception as e:
        logger.error(f"Failed to create chatbot: {str(e)}")
        raise



# https://c19807sl01d04.cloudiax.com:50000/b1s/v1/Login
# A19807_DEMO
# manager
# 12345