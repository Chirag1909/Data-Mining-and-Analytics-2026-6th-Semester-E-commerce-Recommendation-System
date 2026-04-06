from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_transaction_matrix(df_clean, max_baskets=50000):
    """Create basket/transaction matrix from cleaned data"""
    print('Creating transaction matrix...')
    
    # Group by order_id to create baskets
    baskets = df_clean.groupby('order_id')['product_name'].apply(list).reset_index()
    
    # Sample for performance if too large
    if len(baskets) > max_baskets:
        baskets = baskets.sample(n=max_baskets, random_state=42)
    
    transactions = baskets['product_name'].tolist()
    print(f'Created {len(transactions)} baskets')
    
    return transactions

def fit_apriori_model(transactions, min_support=0.005, min_confidence=0.2):
    """Fit Apriori model and generate rules"""
    from mlxtend.preprocessing import TransactionEncoder
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    basket_df = basket_df.replace(False, 0).astype('int')
    
    print(f'Basket DataFrame shape: {basket_df.shape}')
    
    # Apriori frequent itemsets
    frequent_items = apriori(
        basket_df, 
        min_support=min_support, 
        use_colnames=True,
        low_memory=True
    )
    print(f'Found {len(frequent_items)} frequent itemsets')
    
    # Generate rules
    if len(frequent_items) == 0:
        raise ValueError('No frequent itemsets found. Try lower min_support.')
    
    rules = association_rules(
        frequent_items, 
        metric='confidence', 
        min_threshold=min_confidence
    )
    
    # Filter high-quality rules
    rules = rules[rules['lift'] > 1].sort_values('lift', ascending=False)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rules, 'models/apriori_rules.joblib')
    print(f'✅ Saved {len(rules)} rules to models/apriori_rules.joblib')
    
    return rules

def analyze_rules_quality(rules):
    """Create quality analysis plots"""
    rules = rules.copy()
    rules['ante_len'] = rules['antecedents'].apply(len)
    rules['cons_len'] = rules['consequents'].apply(len)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Rule length distribution
    axes[0,0].hist(rules['ante_len'], bins=range(1, max(rules['ante_len'])+2), alpha=0.7, label='Antecedents')
    axes[0,0].hist(rules['cons_len'], bins=range(1, max(rules['cons_len'])+2), alpha=0.7, label='Consequents')
    axes[0,0].set_title('Rule Length Distribution')
    axes[0,0].legend()
    
    # Lift vs Confidence
    scatter = axes[0,1].scatter(rules['confidence'], rules['lift'], c=rules['support'], 
                                s=50, alpha=0.6, cmap='viridis')
    axes[0,1].set_xlabel('Confidence')
    axes[0,1].set_ylabel('Lift')
    axes[0,1].set_title('Lift vs Confidence (colored by Support)')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # Support distribution
    axes[1,0].hist(rules['support']*1000, bins=30, edgecolor='black')
    axes[1,0].set_title('Support Distribution (x1000)')
    axes[1,0].set_xlabel('Support * 1000')
    
    # Top rules table
    top_rules = rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    axes[1,1].axis('tight')
    axes[1,1].axis('off')
    top_table = axes[1,1].table(cellText=top_rules.round(3).values,
                               colLabels=['Antecedents', 'Consequents', 'Support', 'Conf', 'Lift'],
                               cellLoc='center', loc='center')
    top_table.auto_set_font_size(False)
    top_table.set_fontsize(9)
    top_table.scale(1.2, 1.5)
    axes[1,1].set_title('Top 10 Rules')
    
    plt.tight_layout()
    plt.savefig('results/rules_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig
