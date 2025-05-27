# chatbot.py
import os
import re
import json
import logging
import requests
from typing import Dict
from datetime import datetime, timedelta
from sapb1 import SAPB1ServiceLayer

# Configure logging to both console and file
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create a timestamped log file
log_filename = os.path.join(log_directory, f"chatbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to logger
logger.handlers = []  # Clear any existing handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class NLPProcessor:
    """Uses Ollama to extract customer details and order details from user input as JSON."""

    def __init__(self, ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"), 
                 model_name: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")):
        self.ollama_url = ollama_url
        self.model_name = model_name
        logger.info(f"Initialized NLPProcessor with model: {self.model_name}")

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        logger.info(f"Sending request to Ollama with payload: {json.dumps(payload, ensure_ascii=False)}")
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()["response"].strip()
            logger.info(f"Received response from Ollama: {result}")
        
            # Extract JSON between first { and last }
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != -1:
                return result[start:end]
            logger.warning("No valid JSON found in Ollama response, returning empty JSON")
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
            logger.info(f"Parsed and validated JSON: {json.dumps(parsed, ensure_ascii=False)}")
            return parsed
        except Exception as e:
            logger.warning(f"Invalid JSON from Ollama: {result}, Error: {str(e)}")
            default_response = {
                "Email": "",
                "CustomerName": "",
                "CardCode": "",
                "DocDate": doc_date,
                "DocDueDate": doc_due_date,
                "DocumentLines": []
            }
            logger.info(f"Returning default JSON due to parsing error: {json.dumps(default_response, ensure_ascii=False)}")
            return default_response

class Chatbot:
    """Handles user input with customer details and orders, retrieves CardCode, and places orders."""

    def __init__(self, service_layer: SAPB1ServiceLayer, nlp_processor: NLPProcessor):
        self.db = service_layer
        self.nlp = nlp_processor
        logger.info("Initialized Chatbot with SAPB1ServiceLayer and NLPProcessor")

    def _is_arabic_input(self, user_input: str) -> bool:
        is_arabic = any(char in user_input for char in "ابتثجحخدذرزسشصضطظعغفقكلمنهويةى")
        logger.info(f"Checked if input is Arabic: {is_arabic}")
        return is_arabic

    def _generate_fallback_response(self, is_arabic: bool) -> str:
        response = "من فضلك وضح طلبك أكثر." if is_arabic else "Could you please clarify your request?"
        logger.info(f"Generated fallback response: {response}")
        return response

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
                response = "البيانات غير مكتملة. يرجى تضمين الطلبات والتواريخ." if is_arabic else "Incomplete input. Please include orders and dates."
                logger.info(f"Response due to incomplete input: {response}")
                return response

            # Try to retrieve CardCode using priority: input > email > name
            card_code = card_code_input or (
                self.db.get_card_code_by_email(email) if email else (
                    self.db.get_card_code_by_name(customer_name) if customer_name else None
                )
            )

            if not card_code:
                # Use default customer for testing
                default_card_code = "C0001"  # Default CardCode
                logger.info(f"Using default customer CardCode: {default_card_code} as no customer was found")
                card_code = default_card_code

            # Resolve ItemNames to ItemCodes
            resolved_lines = []
            for line in lines:
                item_name = line.get("ItemName")
                quantity = line.get("Quantity")
                item_code = self.db.get_item_code_by_name(item_name)
                if not item_code:
                    response = f"رمز العنصر لـ '{item_name}' غير موجود." if is_arabic else f"Item code for '{item_name}' not found."
                    logger.info(f"Response due to missing item code: {response}")
                    return response
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

            # Log final payload to SAP B1
            logger.info(f"Final payload to SAP B1: {json.dumps(payload, ensure_ascii=False)}")

            # Place order and get success status and order ID
            result = self.db.place_order_from_payload(payload)
            item_summary = ", ".join([f"{line['Quantity']} units of {line['ItemCode']}" for line in resolved_lines])
            response = f"تم تسجيل طلبك لـ {item_summary}." if is_arabic else f"Order placed: {item_summary}."
            logger.info(f"Chatbot response: {response}")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            is_arabic = self._is_arabic_input(user_input)
            response = "حدث خطأ، من فضلك حاول مرة أخرى." if is_arabic else "An error occurred. Please try again."
            logger.info(f"Error response: {response}")
            return response

def create_chatbot():
    try:
        # Get environment variables with proper validation
        base_url = os.getenv("SAPB1_BASE_URL")
        if not base_url:
            logger.error("SAPB1_BASE_URL environment variable is not set")
            raise ValueError("SAPB1_BASE_URL environment variable is not set")
            
        # Convert verify_ssl string to boolean
        verify_ssl_str = os.getenv("SAPB1_VERIFY_SSL", "False")
        verify_ssl = verify_ssl_str.lower() == "true"
        
        service_layer = SAPB1ServiceLayer(
            base_url=base_url,
            company=os.getenv("SAPB1_COMPANY"),
            username=os.getenv("SAPB1_USERNAME"),
            password=os.getenv("SAPB1_PASSWORD"),
            verify_ssl=verify_ssl
        )
        nlp = NLPProcessor()
        chatbot = Chatbot(service_layer, nlp)
        logger.info("Chatbot created successfully")
        return chatbot
    except Exception as e:
        logger.error(f"Failed to create chatbot: {str(e)}")
        raise