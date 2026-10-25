import json
from pathlib import Path

from langchain_core.tools import tool


def load_orders():

    path = Path("data/orders.json")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@tool
def get_order_status(order_id: str) -> str:
    """
    Get the status and details of a customer's order.
    Use this tool when the user asks about a specific order.
    """

    orders = load_orders()

    for order in orders:

        if order["order_id"] == order_id:

            return (
                f"Order ID: {order['order_id']}\n"
                f"Product: {order['product']}\n"
                f"Status: {order['status']}\n"
                f"Expected Delivery: {order['expected_delivery']}\n"
                f"Amount: ₹{order['amount']}\n"
                f"Payment Status: {order['payment_status']}"
            )

    return f"No order found with ID {order_id}."


@tool
def check_refund_eligibility(order_id: str) -> str:
    """
    Check whether an order may be eligible for a refund.
    """

    orders = load_orders()

    for order in orders:

        if order["order_id"] == order_id:

            if order["status"] == "Delivered":

                return (
                    f"Order {order_id} is delivered. "
                    "The customer may request a refund "
                    "if the request is within 30 days "
                    "and the product meets the refund policy."
                )

            if order["status"] == "Cancelled":

                return (
                    f"Order {order_id} was cancelled and "
                    "its payment status is "
                    f"{order['payment_status']}."
                )

            return (
                f"Order {order_id} has status "
                f"{order['status']}. "
                "Refund eligibility should be checked "
                "after reviewing the refund policy."
            )

    return f"No order found with ID {order_id}."