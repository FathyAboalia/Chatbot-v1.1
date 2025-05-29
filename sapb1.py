# sapb1.py

import logging      # لتسجيل الأحداث (نجاح، تحذير، خطأ)
import requests     # لإرسال طلبات HTTP إلى SAP B1
import os           # لإدارة متغيرات البيئة
import re           # لإجراء عمليات البحث والتعامل مع النصوص
import json         # لتحويل البيانات بين JSON و Python
from typing import Optional, Dict   # لتحديد أنواع المتغيرات والوسائط
from datetime import datetime     # لإدارة التواريخ والوقت

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SAPB1ServiceLayer:
    def __init__(self, base_url: str, company: str, username: str, password: str, verify_ssl: bool):
        # Validate base_url to ensure it's properly formatted
        if not base_url:
            raise ValueError("base_url cannot be None or empty")
        
        # Ensure base_url has proper scheme
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            base_url = "https://" + base_url
            
        self.base_url = base_url
        self.company = company
        self.username = username
        self.password = password
        
        # Handle verify_ssl parameter properly
        if isinstance(verify_ssl, str):
            self.verify_ssl = verify_ssl.lower() == "true"
        else:
            self.verify_ssl = bool(verify_ssl)
            
        self.session = requests.Session()
        self.session.verify = self.verify_ssl
        self.session.headers.update({"Content-Type": "application/json"})
        self.logged_in = False
        self.login()

    def login(self):
        login_url = f"{self.base_url}/Login"
        payload = {
            "CompanyDB": self.company,
            "UserName": self.username,
            "Password": self.password
        }
        try:
            response = self.session.post(login_url, json=payload)
            response.raise_for_status()
            data = response.json()
            session_id = data.get("SessionId")
            route_id = data.get("RouteId")

            self.session.headers.update({
                "B1SESSION": session_id,
                "ROUTEID": route_id
            })
            self.logged_in = True
            logger.info("Logged in to SAP B1 Service Layer.")
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            self.logged_in = False

    def request_with_reauth(self, method, url, **kwargs):
        if not self.logged_in:
            self.login()
        full_url = f"{self.base_url}/{url}"
        response = self.session.request(method, full_url, **kwargs)

        if response.status_code == 401:
            logger.warning("Session expired. Logging in again...")
            self.login()
            response = self.session.request(method, full_url, **kwargs)

        response.raise_for_status()
        return response

    def get_card_code_by_email(self, email: str) -> Optional[str]:
        try:
            endpoint = f"BusinessPartners?$filter=EmailAddress eq'{email}'"
            response = self.request_with_reauth("GET", endpoint)
            data = response.json()
            if data["value"]:
                return data["value"][0]["CardCode"]
        except Exception as e:
            logger.error(f"Error retrieving CardCode for email {email}: {str(e)}")
        return None

    def get_card_code_by_name(self, customer_name: str) -> Optional[str]:
        try:
            endpoint = f"BusinessPartners?$filter=CardName eq '{customer_name}'"
            response = self.request_with_reauth("GET", endpoint)
            data = response.json()
            if data["value"]:
                return data["value"][0]["CardCode"]
        except Exception as e:
            logger.error(f"Error retrieving CardCode for name {customer_name}: {str(e)}")
        return None

    def get_item_code_by_name(self, item_name: str) -> Optional[str]:
        try:
            endpoint = f"Items?$filter=ItemName eq '{item_name}'"
            response = self.request_with_reauth("GET", endpoint)
            data = response.json()
            if data["value"]:
                return data["value"][0]["ItemCode"]
        except Exception as e:
            logger.error(f"Error retrieving item code for {item_name}: {str(e)}")
        return None

    def place_order_from_payload(self, payload: Dict) -> bool:
        try:
            response = self.request_with_reauth("POST", "Orders", json=payload)
            data = response.json()
            order_id = data.get("DocEntry")  # Extract DocEntry from response
            logger.info(f"Order placed successfully with payload: {json.dumps(payload, ensure_ascii=False)}")
            return True
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False