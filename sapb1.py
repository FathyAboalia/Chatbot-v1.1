import requests
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAPB1ServiceLayer:
    def __init__(self, base_url: str, company: str, username: str, password: str, verify_ssl: bool = True):
        self.base_url = base_url
        self.company = company
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.verify = verify_ssl  # SSL verification toggle
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

    def get_price(self, product_name: str) -> Optional[float]:
        try:
            url = f"{self.base_url}/Items?$filter=contains(ItemName,'{product_name}')&$select=ItemCode,ItemName,ItemPrices"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get("value", [])
            return float(items[0]["ItemPrices"][0]["Price"]) if items else None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get price for {product_name}: {str(e)}")
            raise Exception(f"Failed to get price: {str(e)}")

    def check_stock(self, product_name: str) -> int:
        try:
            url = f"{self.base_url}/Items?$filter=contains(ItemName,'{product_name}')&$select=ItemCode,ItemWarehouseInfoCollection"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get("value", [])
            return sum(wh.get("InStock", 0) for wh in items[0].get("ItemWarehouseInfoCollection", [])) if items else 0
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check stock for {product_name}: {str(e)}")
            raise Exception(f"Failed to check stock: {str(e)}")

    def place_order(self, product_name: str, quantity: int) -> str:
        try:
            url = f"{self.base_url}/Items?$filter=contains(ItemName,'{product_name}')&$select=ItemCode"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get("value", [])
            if not items:
                return f"No product named {product_name} found."
            item_code = items[0]["ItemCode"]

            order_url = f"{self.base_url}/Orders"
            payload = {                 
                "CardCode": "C20000",  # Replace with actual customer code if needed
                "DocumentLines": [{"ItemCode": item_code, "Quantity": quantity}]
            }
            response = self.session.post(order_url, json=payload)
            if response.status_code == 201:
                logger.info(f"Order placed: {quantity} units of {product_name}")
                return f"Order for {quantity} units of {product_name} placed successfully."
            return "Error placing order."
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