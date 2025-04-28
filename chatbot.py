# chatbot.py

import re
import os
import logging
from typing import Tuple, Optional
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self):
        try:
            # Use PyTorch explicitly by setting framework="pt"
            self.classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased", framework="pt")
            logger.info("NLP Processor initialized successfully with PyTorch.")
        except Exception as e:
            logger.error(f"Failed to initialize NLP Processor: {str(e)}")
            raise

    def process_input(self, user_input: str) -> Tuple[str, Optional[str], int]:
        user_input_lower = user_input  # بدون .lower()

        # تحديد الـ intent بتحويل لصغير بس في المقارنة
        INTENTS = ["place_order", "check_stock"]
        if any(kw in user_input_lower.lower() for kw in ["order", "buy", "place"]):
            intent = "place_order"
        else:
            result = self.classifier(user_input, candidate_labels=INTENTS)
            intent = result["labels"][0]

        # استخراج الكمية
        quantity_match = re.search(r"(\d+)", user_input_lower)
        quantity = int(quantity_match.group(0)) if quantity_match else 1

        # الكلمات اللي هنشيلها
        keywords = ["order", "buy", "place"]
        for word in keywords:
            user_input_lower = user_input_lower.replace(word, " ").replace(word.lower(), " ")

        # تنظيف النص واستخراج المنتج
        product = " ".join(user_input_lower.split()).strip()
        if quantity_match:
            product = product.replace(quantity_match.group(0), "").strip()

        logger.info(f"المنتج المستخرج: {product}")

        if not product:
            return intent, None, quantity

        return intent, product, quantity

class Chatbot:
    def __init__(self, service_layer, nlp_processor: NLPProcessor):
        self.db = service_layer
        self.nlp = nlp_processor

    def get_response(self, user_input: str) -> str:
        try:
            intent, product, quantity = self.nlp.process_input(user_input)

            if not product:
                return "من فضلك حدد اسم المنتج." if "سعر" in user_input.lower() else "Please specify the product name."

            elif intent == "place_order":
                return self.db.place_order(product, quantity)

            elif intent == "check_stock":
                stock = self.db.check_stock(product)
                return f"مخزون {product} حالياً هو {stock} وحدة." if "مخزون" in user_input.lower() else f"Current stock of {product} is {stock} units."

            return "مش فاهم طلبك، ممكن توضح؟" if "سعر" in user_input.lower() else "I didn't understand your request. Could you clarify?"
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise

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