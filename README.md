🛒 E-commerce Recommendation System
📌 Overview
This project implements a hybrid recommendation system using the Instacart dataset. It combines K-Means clustering, Apriori association rules, and content-based filtering to generate personalized product recommendations.

🚀 Features
📊 Data preprocessing & feature engineering
👥 Customer segmentation using K-Means
🛍️ Association rule mining (Support, Confidence, Lift)
🔗 Hybrid recommendation approach
🌐 Interactive UI using Gradio

🧠 Tech Stack
Python, Pandas, NumPy
Scikit-learn (Clustering)
MLxtend (Apriori)
Matplotlib, Seaborn
Gradio (Web Interface)

📂 Project Structure
├── app/
│   └── app.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── apriori_model.py
│   └── recommender.py
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_clustering.ipynb
│   ├── 4_apriori.ipynb
│   └── 5_recommendation.ipynb
├── requirements.txt
└── README.md

📊 Results
🔹 Top Products
🔹 Clustering (Elbow Method)
🔹 Association Rules

🌐 Demo (Gradio UI)
Run locally:
python app/app.py
Then open the generated link to interact with the system.

📥 Dataset
Dataset not included due to size limitations.
Download from:
👉 https://www.instacart.com/datasets/grocery-shopping-2017

⚙️ Installation
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

🎯 Objective
To build a scalable recommendation system that enhances user experience and improves product discovery in e-commerce platforms.

👨‍💻 Author
Chirag Khodiyar, Sayan Mitra
