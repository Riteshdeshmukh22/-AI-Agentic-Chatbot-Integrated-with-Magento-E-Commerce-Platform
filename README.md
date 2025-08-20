**🛒 Agentic E-commerce Chatbot with Cart Functionality**

**📌 Overview**
  This project is an AI-powered agentic chatbot designed for an e-commerce platform (Magento demo data).
  It enables customers to:
    ~Search products, get recommendations, and view product details.
    ~Add, view, and remove items from the shopping cart.
    ~Checkout with a price summary.
  The chatbot uses LLMs (Gemini) combined with vector search (FAISS + Sentence Transformers) to ensure responses are grounded only in product data (RAG – Retrieval-Augmented Generation).
  The RAG implementaion restrict the bot to Answer the user query only from the provided data.
  
**🛠️ Technologies Used**
  ~Python
  ~LangChain (for LLM orchestration)
  ~SentenceTransformers (all-MiniLM-L6-v2) – For product embeddings
  ~FAISS – Vector similarity search
  ~Google Generative AI (Gemini) – LLM for conversational responses
  ~NumPy – Embedding handling
  ~JSON – Product database storage

**📚 Key Libraries**
  ~sentence-transformers
  ~faiss
  ~langchain-google-genai
  ~numpy
  ~json
  ~os
  ~
**🤖 AI / ML Concepts**

  1.Embeddings → Product data (name, description, category, price) converted into vectors.
  2.FAISS Indexing → Efficient similarity search to recommend relevant products.
  3.RAG (Retrieval-Augmented Generation) → LLM answers user queries only from product data.
  4.LLM (Gemini 2.0 Flash) → Provides natural, contextual chatbot responses.
  5.Cart Management → Add, view, remove, and checkout items in a conversational way.

**📑 Features**
✅ Query products by name, category, or description
✅ Get relevant product recommendations
✅ Add products to cart (add product_name to cart)
✅ Show or remove items from cart
✅ Fallback to “Sorry, I don’t have that information” if product not in data.
