from src.data_loader import load_data
from src.preprocessing import preprocess
from src.clustering import cluster_users
from src.apriori_model import create_basket, generate_rules
from src.recommender import recommend

def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    train, test = preprocess(df)

    print("Clustering users...")
    cluster_users(train)

    print("Generating recommendation model...")
    basket = create_basket(train)
    rules = generate_rules(basket)

    print("\nSample products you can try:")
    print(train['product_name'].value_counts().head(10))

    print("\n--- PRODUCT RECOMMENDATION SYSTEM ---")

    while True:
        product = input("\nEnter product name (or 'exit'): ")

        if product.lower() == 'exit':
            break

        recs = recommend(product, rules, train)

        print("Recommended products:", recs[:5])


if __name__ == "__main__":
    main()