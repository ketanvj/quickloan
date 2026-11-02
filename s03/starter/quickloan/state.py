from typing import TypedDict


class QuickLoanState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]
    query_type:       str
