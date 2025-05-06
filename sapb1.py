# sapb1.py

import logging
import requests
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SAPB1ServiceLayer:
    def __init__(self, base_url: str, company: str, username: str, password: str, verify_ssl: bool = False):
        self.base_url = base_url
        self.company = company
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.verify = verify_ssl  # Disable SSL verification for testing; enable in production
        self._login()

    def _login(self):
        try:
            login_url = f"{self.base_url}/Login"
            payload = {"CompanyDB": self.company, "UserName": self.username, "Password": self.password}
            response = self.session.post(login_url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully logged into SAP B1 at {self.base_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Login failed: {str(e)}")
            raise Exception(f"Failed to login to SAP B1 Service Layer: {str(e)}")

    def place_order_from_payload(self, payload: dict) -> str:
        """Send full JSON payload to SAP B1 Orders API."""
        try:
            order_url = f"{self.base_url}/Orders"
            response = self.session.post(order_url, json=payload)
            if response.status_code == 201:
                logger.info("Order placed successfully.")
                return "Order placed successfully."
            logger.error(f"Failed to place order: {response.text}")
            return f"Error placing order: {response.text}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise Exception(f"Failed to place order: {str(e)}")

    def place_order(self, product_name: str, quantity: int, card_code: str = "C0001") -> str:
        try:
            # Use exact match for ItemName
            url = f"{self.base_url}/Items?$filter=ItemName eq '{product_name}'&$select=ItemCode"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get("value", [])
            if not items:
                logger.warning(f"No product named '{product_name}' found in SAP B1")
                return f"No product named {product_name} found."
            item_code = items[0]["ItemCode"]

            order_url = f"{self.base_url}/Orders"
            payload = {
                "CardCode": card_code,
                "DocDueDate": "2025-05-05",  # Consider making this dynamic
                "DocumentLines": [{"ItemCode": item_code, "Quantity": quantity, "BPLId": 1}]  # BPLId may need configuration
            }
            response = self.session.post(order_url, json=payload)
            if response.status_code == 201:
                logger.info(f"Order placed: {quantity} units of {product_name}")
                return f"Order for {quantity} units of {product_name} placed successfully."
            logger.error(f"Failed to place order: {response.text}")
            return f"Error placing order: {response.text}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to place order for {product_name}: {str(e)}")
            raise Exception(f"Failed to place order: {str(e)}")

    def close(self):
        try:
            logout_url = f"{self.base_url}/Logout"
            self.session.post(logout_url)
            self.session.close()
            logger.info("SAP B1 session closed.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to logout: {str(e)}")