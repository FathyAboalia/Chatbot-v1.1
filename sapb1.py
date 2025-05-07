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
        self.session.verify = verify_ssl
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

    def get_card_code_by_email(self, email: str) -> Optional[str]:
        """Retrieve CardCode from SAP B1 based on email address."""
        try:
            url = f"{self.base_url}/BusinessPartners?$filter=EmailAddress eq '{email}'&$select=CardCode"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            partners = data.get("value", [])
            if not partners:
                logger.warning(f"No business partner found for email: {email}")
                return None
            return partners[0]["CardCode"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve CardCode for email {email}: {str(e)}")
            return None

    def get_card_code_by_name(self, customer_name: str) -> Optional[str]:
        """Retrieve CardCode from SAP B1 based on customer name."""
        try:
            url = f"{self.base_url}/BusinessPartners?$filter=CardName eq '{customer_name}'&$select=CardCode"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            partners = data.get("value", [])
            if not partners:
                logger.warning(f"No business partner found for customer name: {customer_name}")
                return None
            return partners[0]["CardCode"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve CardCode for customer name {customer_name}: {str(e)}")
            return None

    def validate_item_code(self, item_code: str) -> bool:
        """Validate if an ItemCode exists in SAP B1."""
        try:
            url = f"{self.base_url}/Items?$filter=ItemCode eq '{item_code}'&$select=ItemCode"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get("value", [])
            return bool(items)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to validate ItemCode {item_code}: {str(e)}")
            return False

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
                "DocDate": datetime.now().strftime("%Y-%m-%d"),
                "DocDueDate": datetime.now().strftime("%Y-%m-%d"),
                "DocumentLines": [{"ItemCode": item_code, "Quantity": quantity, "BPLId": 1}]
            }
            response = self.session.post(order_url, json=payload)
            if response.status_code == 201:
                logger.info(f"Order placed: {quantity} units of {product_name}")
                return f"Order for {quantity} units of {product_name} placed successfully."
            logger.error(f"Failed to place order: {response.text}")
            return f"Error placing order: {response.text}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to place order for {product_name}: {str(e)}")
            return f"Failed to place order: {str(e)}"

    def close(self):
        try:
            logout_url = f"{self.base_url}/Logout"
            self.session.post(logout_url)
            self.session.close()
            logger.info("SAP B1 session closed.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to logout: {str(e)}")