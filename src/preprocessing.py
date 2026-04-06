def preprocess(df):
    df = df[['user_id', 'order_id', 'product_id', 'product_name']]

    # Increase data size
    df = df.sample(n=200000, random_state=42)

    # Increase products (IMPORTANT)
    top_products = df['product_name'].value_counts().head(800).index
    df = df[df['product_name'].isin(top_products)]

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test