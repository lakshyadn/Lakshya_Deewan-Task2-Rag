✈️ RAG-Based Chatbot for Aurora Skies Airways

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers frequently asked questions (FAQs) for Aurora Skies Airways.
It retrieves relevant information from the provided FAQ dataset and generates accurate, context-grounded responses using an open-source language model.

🎯 Objective

To design and build a chatbot capable of:

Retrieving the most relevant FAQ entries for a given customer query

Generating a precise and natural response using a pre-trained LLM

Reducing hallucinations by grounding answers in the actual FAQ data

🧰 Technologies Used

Python 3.8+

Pandas – Data loading and preprocessing

Sentence Transformers – For text embeddings

FAISS – Vector similarity search

Transformers (Hugging Face) – Text generation

NumPy / Pickle – For array handling and storage

Screenshot:
![image alt](https://github.com/lakshyadn/Lakshya_Deewan-Task2-Rag/blob/2a77993a639c7c8b13433432ef38b3df3d25140a/Screenshot%202025-10-31%20011552.png)
![image alt](https://github.com/lakshyadn/Lakshya_Deewan-Task2-Rag/blob/5d8f0e33bbe2436b728917b32e9e5f3bf4066c7b/image.png)

📂 Dataset

File Name: airline_faq.csv

Columns:

Question,Answer
What is the baggage allowance?,Each passenger is allowed one carry-on and one checked bag up to 20kg.
How can I reschedule my flight?,You can reschedule via our website or mobile app under 'Manage Booking'.
...

⚙️ Setup and Execution
🖥️ Run Locally (VS Code / Terminal)

Install dependencies

pip install pandas sentence-transformers faiss-cpu transformers numpy


Place your dataset (airline_faq.csv) in the same folder as your Python file or notebook.

Run the script or notebook

python rag_chatbot.py


or open the notebook in Jupyter/Colab and run all cells step-by-step.

Ask a question
Example input:

What is the refund policy?


Example output:

You can request a refund through our website’s “Manage Booking” section.
