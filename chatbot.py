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
            result = response.json()["response"].strip()
        
            # Extract JSON between first { and last }
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != -1:
                return result[start:end]
            return "{}"
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            return "{}"

    def extract_structured_json(self, user_input: str) -> dict:
        # Use dynamic dates for consistency
        doc_date = datetime.now().strftime("%Y-%m-%d")
        doc_due_date = datetime.now().strftime("%Y-%m-%d")

        prompt = (
            f'Analyze the user input: "{user_input}". Extract: '
            f'- Email address (e.g., test@example.com). '
            f'- Customer name (e.g., Ahmed Fathy). '
            f'- CardCode (e.g., C12345) if explicitly mentioned. '
            f'- Orders, each with an item name (e.g., Test Item) and quantity (e.g., 5). '
            f'The input may be formatted as: '
            f'- Direct order: "order 5 Test Item for test@example.com" '
            f'- Conversational: "test@example.com could you order 5 Test Item" '
            f'- Email-like: "From: test@example.com, Subject: Order, Please order 5 Test Item" '
            f'- Email reference: "Process the order from test@example.com for 5 Test Item" '
            f'- Multiple items: "order 5 Test Item and 2 Other Item for test@example.com" '
            f'Return a single JSON object: '
            f'{{"Email": "<email_address>", "CustomerName": "<customer_name>", "CardCode": "<card_code>", '
            f'"DocDate": "{doc_date}", "DocDueDate": "{doc_due_date}", '
            f'"DocumentLines": [{{"ItemName": "<item_name>", "Quantity": <quantity>}}, ...]}} '
            f'Rules: '
            f'- Use "" for Email, CustomerName, or CardCode if not found. '
            f'- Use [] for DocumentLines if no orders are identified. '
            f'- Preserve item name text (e.g., "Test Item"), removing quotes if present. '
            f'- Quantities must be positive integers; ignore invalid ones. '
            f'- Ignore irrelevant text (e.g., greetings, "could you"). '
            f'Response MUST be a JSON object enclosed in {{}}, with no extra text.'
        )

        result = self.generate_text(prompt)
        try:
            parsed = json.loads(result)
            # Validate DocumentLines
            parsed["DocumentLines"] = [
                line for line in parsed.get("DocumentLines", [])
                if line.get("ItemName") and isinstance(line.get("Quantity"), int) and line["Quantity"] > 0
            ]
            return parsed
        except Exception as e:
            logger.warning(f"Invalid JSON from Ollama: {result}, Error: {str(e)}")
            return {
                "Email": "",
                "CustomerName": "",
                "CardCode": "",
                "DocDate": doc_date,
                "DocDueDate": doc_due_date,
                "DocumentLines": []
            }

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
            # Log user input
            logger.info(f"User input: {user_input}")
            
            structured = self.nlp.extract_structured_json(user_input)
            
            # Log extracted JSON from NLPProcessor
            logger.info(f"Extracted JSON from NLPProcessor: {json.dumps(structured, ensure_ascii=False)}")
            
            email = structured.get("Email", "")
            customer_name = structured.get("CustomerName", "")
            card_code_input = structured.get("CardCode", "")
            lines = structured.get("DocumentLines", [])
            doc_date = structured.get("DocDate")
            doc_due_date = structured.get("DocDueDate")

            is_arabic = self._is_arabic_input(user_input)

            if not lines or not doc_date or not doc_due_date:
                return "البيانات غير مكتملة. يرجى تضمين الطلبات والتواريخ." if is_arabic else "Incomplete input. Please include orders and dates."

            # Try to retrieve CardCode using priority: input > email > name
            card_code = card_code_input or (
                self.db.get_card_code_by_email(email) if email else (
                    self.db.get_card_code_by_name(customer_name) if customer_name else None
                )
            )

            if not card_code:
                # إذا لم يتم العثور على العميل، استخدم عميل افتراضي للتجربة
                default_card_code = "C00001"  # رمز العميل الافتراضي
                logger.info(f"Using default customer CardCode: {default_card_code} as no customer was found")
                card_code = default_card_code
                # return "لم يتم العثور على عميل بناءً على المعلومات المقدمة." if is_arabic else "No customer found based on the provided information."
                # تم تعليق السطر السابق واستبداله بالعميل الافتراضي

            # Resolve ItemNames to ItemCodes
            resolved_lines = []
            for line in lines:
                item_name = line.get("ItemName")
                quantity = line.get("Quantity")
                item_code = self.db.get_item_code_by_name(item_name)
                if not item_code:
                    return f"رمز العنصر لـ '{item_name}' غير موجود." if is_arabic else f"Item code for '{item_name}' not found."
                resolved_lines.append({"ItemCode": item_code, "Quantity": quantity})

            # Log resolved CardCode and DocumentLines
            logger.info(f"Resolved CardCode: {card_code}, DocumentLines: {json.dumps(resolved_lines, ensure_ascii=False)}")

            # Construct JSON with CardCode and resolved ItemCodes
            payload = {
                "CardCode": card_code,
                "DocDate": doc_date,
                "DocDueDate": doc_due_date,
                "DocumentLines": resolved_lines
            }

            # Log final payload
            logger.info(f"Final payload to SAP B1: {json.dumps(payload, ensure_ascii=False)}")

            result = self.db.place_order_from_payload(payload)
            item_summary = ", ".join([f"{line['Quantity']} units of {line['ItemCode']}" for line in resolved_lines])
            return f"تم تسجيل طلبك لـ {item_summary}." if is_arabic else f"Order placed: {item_summary}."

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

