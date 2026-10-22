from tools.product_tools import search_products

products = search_products.invoke({
    "search": "hoodie"
})

for product in products:
    print(product)