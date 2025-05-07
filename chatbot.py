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
    """Uses Ollama to extract customer details and order details from user input as JSON."""

    def __init__(self, ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"), 
                 model_name: str = os.getenv("OLLAMA_MODEL", "llama3:instruct")):
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
        doc_date = datetime.now().strftime("%Y-%m-%d")
        doc_due_date = datetime.now().strftime("%Y-%m-%d")

        prompt = (
            f'Given the input: "{user_input}", extract the following: '
            f'- The email address (e.g., customer@example.com) if present. '
            f'- The customer name (e.g., Ahmed, Fathy) if present. '
            f'- The CardCode if present. '
            f'- A list of orders, where each order includes an item code and quantity. '
            f'Respond ONLY with a single raw JSON object in the following format: '
            f'{{"Email": "<email_address>", "CustomerName": "<customer_name>", "CardCode": "<card_code>", '
            f'"DocDate": "{doc_date}", "DocDueDate": "{doc_due_date}", '
            f'"DocumentLines": [{{"ItemCode": "<item_code>", "Quantity": <quantity>}}, ...]}} '
            f'If a field is not present, use an empty string for strings or an empty list for DocumentLines. '
            f'Your entire response MUST start and end with {{}} and contain nothing else.'
        )

        result = self.generate_text(prompt)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"Invalid JSON from Ollama: {result}")
            return {}

class Chatbot:
    """Handles user input with customer details and orders, retrieves CardCode, and places orders."""

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
            email = structured.get("Email", "")
            customer_name = structured.get("CustomerName", "")
            card_code_input = structured.get("CardCode", "")
            lines = structured.get("DocumentLines", [])
            doc_date = structured.get("DocDate")
            doc_due_date = structured.get("DocDueDate")

            is_arabic = self._is_arabic_input(user_input)

            if not lines or not doc_date or not doc_due_date:
                return "البيانات غير مكتملة. يرجى تضمين الطلبات والتواريخ." if is_arabic else "Incomplete input. Please include orders and dates."

            # Try to retrieve CardCode based on available identifiers
            card_code = None
            if card_code_input:
                card_code = card_code_input  # Use CardCode if provided
            elif email:
                card_code = self.db.get_card_code_by_email(email)
            elif customer_name:
                card_code = self.db.get_card_code_by_name(customer_name)

            if not card_code:
                return "لم يتم العثور على عميل بناءً على المعلومات المقدمة." if is_arabic else "No customer found based on the provided information."

            # Validate ItemCodes
            for line in lines:
                item_code = line.get("ItemCode")
                if not item_code or not self.db.validate_item_code(item_code):
                    return f"رمز العنصر {item_code} غير صالح أو غير موجود." if is_arabic else f"Item code {item_code} is invalid or not found."

            # Construct JSON with CardCode
            payload = {
                "CardCode": card_code,
                "DocDate": doc_date,
                "DocDueDate": doc_due_date,
                "DocumentLines": lines
            }

            result = self.db.place_order_from_payload(payload)
            item_summary = ", ".join([f"{line['Quantity']} units of {line['ItemCode']}" for line in lines])
            return f"تم تسجيل طلبك لـ {item_summary}." if is_arabic else f"Order placed: {item_summary}."

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            is_arabic = self._is_arabic_input(user_input)
            return "حدث خطأ، من فضلك حاول مرة أخرى." if is_arabic else "An error occurred. Please try again."

def create_chatbot():
    try:
        service_layer = SAPB1ServiceLayer(
            base_url=os.getenv("SAPB1_BASE_URL", "https://c19807sl01d04.cloudiax.com:50000/b1s/v1"),
            company=os.getenv("SAPB1_COMPANY", "BS_PRODUCTIVE"),
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