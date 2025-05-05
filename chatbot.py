# chatbot.py

import os
import re
import logging
from typing import Tuple, Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Processes user input using a fine-tuned Flan-T5 for intent detection and entity extraction."""
    
    def __init__(self, model_path: str = "./flan-t5-finetuned"):
        """Initialize fine-tuned Flan-T5 model and tokenizer."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
            logger.info(f"Initialized fine-tuned Flan-T5 from {model_path} on {self.device}")

            # Load product names for prompt
            self.products = [
                "Farmer", "Accountant", "Teacher", "Engineer", "Credit Hour Engineering 2023",
                "Keratin Shampoo 1 L", "Keratin Shampoo 500 ML", "Zakhir Perfume oil 100 GMS",
                "new Parent Item", "فاصوليا خام", "Mobile Samsung 8 Gega", "فاصوليا مجمده(نص مصنع)",
                "New Child Item", "شكاره فاصوليا 20ك", "Item1", "شكاره", "شكاره فاصوليا 25ك",
                "Office Chair With a stand", "Item2"
            ]
        except Exception as e:
            logger.error(f"Failed to initialize Flan-T5: {str(e)}")
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
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            logger.debug(f"Flan-T5 raw output: {result}")
            return result
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    def detect_intent(self, user_input: str) -> str:
        """Detect the intent of user input using Flan-T5."""
        prompt = (
            f"Determine the intent of the following user input. "
            f"Return a concise intent label: 'place_order', 'check_price', or 'unknown'. "
            f"Input: '{user_input}'"
        )
        intent = self.generate_text(prompt, max_length=20)
        
        # Normalize intent
        intent = intent.lower().replace(" ", "_")
        if not intent or intent == "none":
            intent = "unknown"
        elif intent.startswith(("order", "buy", "purchase", "to_order")):
            intent = "place_order"
            
        return intent

    def extract_entities(self, user_input: str) -> Tuple[Optional[str], int]:
        """Extract product name and quantity from user input using Flan-T5."""
        prompt = (
            f"Extract the product name and quantity from the following input. "
            f"Return in the format: 'product: <name>, quantity: <number>'. "
            f"Available products: {', '.join(self.products)}. "
            f"Match the product name exactly as listed. "
            f"If no product or quantity is found, return 'product: None, quantity: 1'. "
            f"Input: '{user_input}'"
        )
        result = self.generate_text(prompt)
        
        try:
            # More flexible regex to handle various output formats
            product_match = re.search(r"product:\s*([^,]+?)\s*(?:,\s*quantity:|$)", result, re.UNICODE)
            quantity_match = re.search(r"quantity:\s*(\d+)", result, re.UNICODE)
            product = product_match.group(1).strip() if product_match and product_match.group(1).strip() != "None" else None
            quantity = int(quantity_match.group(1)) if quantity_match else 1

            # Validate product against known list
            if product and product not in self.products:
                logger.warning(f"Extracted product '{product}' not in known products")
                product = None
        except Exception as e:
            logger.warning(f"Failed to parse extraction result: {result}. Error: {str(e)}")
            product, quantity = None, 1
            
        return product, quantity

    def process_input(self, user_input: str) -> Tuple[str, Optional[str], int]:
        """Process user input to extract intent, product, and quantity."""
        intent = self.detect_intent(user_input)
        product, quantity = self.extract_entities(user_input)
        
        logger.info(f"Processed input - Intent: {intent}, Product: {product}, Quantity: {quantity}")
        return intent, product, quantity

class Chatbot:
    """Handles user interactions and queries the service layer for responses."""
    
    def __init__(self, service_layer, nlp_processor: NLPProcessor):
        """Initialize chatbot with service layer and NLP processor."""
        self.db = service_layer
        self.nlp = nlp_processor

    def _is_arabic_input(self, user_input: str) -> bool:
        """Detect if the input contains Arabic characters."""
        return any(char in user_input for char in "ابتثجحخدذرزسشصضطظعغفقكلمنهويةى")

    def _generate_fallback_response(self, user_input: str, is_arabic: bool) -> str:
        """Generate a fallback response for unsupported intents."""
        prompt = (
            f"Generate a polite response in {'Arabic' if is_arabic else 'English'} for a chatbot "
            f"when the user's request is unclear or not supported. "
            f"Suggest placing an order if relevant. Input: '{user_input}'"
        )
        return self.nlp.generate_text(prompt)

    def get_response(self, user_input: str) -> str:
        """Generate a response based on user input and service layer results."""
        try:
            intent, product, quantity = self.nlp.process_input(user_input)
            is_arabic = self._is_arabic_input(user_input)

            if not product:
                return "من فضلك حدد اسم المنتج." if is_arabic else "Please specify the product name."

            if intent == "place_order":
                result = self.db.place_order(product, quantity)
                return f"تم تسجيل طلبك لـ {quantity} وحدة من {product}." if is_arabic else result

            return self._generate_fallback_response(user_input, is_arabic)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            is_arabic = self._is_arabic_input(user_input)
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