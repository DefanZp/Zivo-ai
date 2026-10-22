from tools.product_tools import increase_quantity

result = increase_quantity.invoke({
    "product_id": 74
})

print(result)