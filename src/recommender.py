def recommend(product_name, rules, df):
    product_name = product_name.lower()
    recommendations = set()

    for _, row in rules.iterrows():
        antecedents = [str(i).lower() for i in row['antecedents']]
        consequents = [str(i) for i in row['consequents']]

        if any(product_name in item for item in antecedents):
            recommendations.update(consequents)

    # 🔥 fallback if no rules found
    if not recommendations:
        print("⚠️ No strong rules found, showing popular items instead")
        return df['product_name'].value_counts().head(5).index.tolist()

    return list(recommendations)