# chatbot.py

import os
import re
import json
import logging
import requests
from typing import Dict
from datetime import datetime, timedelta
from sapb1 import SAPB1ServiceLayer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Uses Ollama to classify document type and extract structured data as JSON only."""

    def __init__(self, ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"), 
                 model_name: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()["response"]

            # Extract raw JSON only
            match = re.search(r"\{.*\}", result, re.DOTALL)
            return match.group(0).strip() if match else "{}"
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            return "{}"

    def extract_structured_json(self, user_input: str) -> dict:
        # Use dynamic dates for consistency
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.0000000+03:00")
        due = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.0000000+03:00")

        prompt = (
            f'Given the input string: "{user_input}", classify the DocumentType as one of: "Order", "Quotation", or "Price". '
            f'Then, extract entities in the following format: '
            f'{{"DocumentType": "<classified_type>", "CardCode": "<customer_code>", '
            f'"DocumentDate": "{now}", "DocDueDate": "{due}", "DocumentLines": [{{"ItemCode": "<item_code>", "Quantity": <quantity>}}]}} '
            f'Respond ONLY with a single raw JSON object. Do not add any explanations, greetings, or commentary before or after the JSON. '
            f'Your entire response MUST start and end with {{}} and contain nothing else.'
        )

        result = self.generate_text(prompt)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"Invalid JSON from Ollama: {result}")
            return {}

class Chatbot:
    """Handles user interactions and queries the service layer for responses."""

    def __init__(self, service_layer: SAPB1ServiceLayer, nlp_processor: NLPProcessor):
        self.db = service_layer
        self.nlp = nlp_processor

    def _is_arabic_input(self, user_input: str) -> bool:
        return any(char in user_input for char in "ابتثجحخدذرزسشصضطظعغفقكلمنهويةى")

    def _generate_fallback_response(self, is_arabic: bool) -> str:
        return "من فضلك وضح طلبك أكثر." if is_arabic else "Could you please clarify your request?"

    def get_response(self, user_input: str) -> str:
        try:
            structured = self.nlp.extract_structured_json(user_input)
            card = structured.get("CardCode")
            lines = structured.get("DocumentLines", [])
            intent = structured.get("DocumentType", "").lower()
            document_date = structured.get("DocumentDate")
            doc_due_date = structured.get("DocDueDate")

            is_arabic = self._is_arabic_input(user_input)

            if not card or not lines or not document_date or not doc_due_date:
                return "البيانات غير مكتملة." if is_arabic else "Incomplete input."

            if intent == "order":
                # Construct payload directly from structured JSON
                payload = {
                    "CardCode": card,
                    "DocumentDate": document_date,
                    "DocDueDate": doc_due_date,
                    "DocumentLines": lines
                }
                result = self.db.place_order_from_payload(payload)
                item_summary = ", ".join([f"{line['Quantity']} units of {line['ItemCode']}" for line in lines])
                return f"تم تسجيل طلبك لـ {item_summary}." if is_arabic else f"Order placed: {item_summary}."

            return self._generate_fallback_response(is_arabic)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            is_arabic = self._is_arabic_input(user_input)
            return "حدث خطأ، من فضلك حاول مرة أخرى." if is_arabic else "An error occurred. Please try again."

def create_chatbot():
    try:
        service_layer = SAPB1ServiceLayer(
            base_url=os.getenv("SAPB1_BASE_URL", "https://c19807sl01d04.cloudiax.com:50000/b1s/v1"),
            company=os.getenv("SAPB1_COMPANY", "BS_PRODUCTIVE"), # A19807_DEMO
            username=os.getenv("SAPB1_USERNAME", "manager"),
            password=os.getenv("SAPB1_PASSWORD", "12345"),
            verify_ssl=os.getenv("SAPB1_VERIFY_SSL", "True").lower() == "true"
        )
        nlp = NLPProcessor()
        chatbot = Chatbot(service_layer, nlp)
        return chatbot
    except Exception as e:
        logger.error(f"Failed to create chatbot: {str(e)}")
        raise