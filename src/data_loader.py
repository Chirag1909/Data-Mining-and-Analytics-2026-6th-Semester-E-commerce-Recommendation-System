import pandas as pd

def load_data():
    orders = pd.read_csv("data/orders.csv")
    order_products = pd.read_csv("data/order_products__prior.csv")
    products = pd.read_csv("data/products.csv")

    df = order_products.merge(orders, on="order_id")
    df = df.merge(products, on="product_id")

    return df