from tools.product_tools import add_to_cart


result = add_to_cart.invoke({
    "product_id": 74,
    "quantity": 1
})

print(result)