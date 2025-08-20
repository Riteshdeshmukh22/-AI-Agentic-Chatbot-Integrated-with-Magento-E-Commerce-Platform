import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "ZAIzaSmyAvnuR6mw-P-toxrMjb"

def load_products(file_path = "/home/stark/Desktop/ai practice/magento/e_com_data.json"):
    with open("/home/stark/Desktop/ai practice/magento/e_com_data.json", 'r') as f:
        return json.load(f)

def preprocess_product_data(products):
    # Include all fields in the product string for embedding
    return [
        f"Name: {p['name']}. Description: {p['description']}. Category: {p['category']}. Price: {p['price']}. Product ID: {p['product_id']}"
        for p in products
    ]

def generate_embeddings(texts, model):
    return np.array(model.encode(texts)).astype(np.float32)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # Pass embeddings as argument
    return index

def query_faiss_index(index, model, query, k=3):
    query_emb = np.array(model.encode([query])).astype(np.float32)
    D, I = index.search(query_emb, k)
    return I[0]

def recommend_products(user_query, products, model, index, k=3):
    indices = query_faiss_index(index, model, user_query, k)
    recommendations = [products[i] for i in indices]
    return recommendations

def generate_llm_response(user_query, context, llm):
    prompt = (
        f"User: {user_query}\n"
        f"Product context (from database):\n{context}\n"
        "As an e-commerce assistant, answer the user's question using ONLY the above product data. "
        "If the answer is not present in the product data, say 'Sorry, I do not have that information.' "
        "Do NOT make up or infer any details that are not explicitly listed above. "
        "If the user asks for a price or details, be specific and only use the data shown. "
        "Give user recommendations, suggest only the products related to users product."
        "If user asked for any specific category give him proper list"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def format_product_details(product):
    return (
        f"Name: {product['name']}\n"
        f"Description: {product['description']}\n"
        f"Category: {product['category']}\n"
        f"Price: ${product['price']}\n"
        f"Product ID: {product['product_id']}"
    )

if __name__ == "__main__":
    products = load_products("/home/stark/Desktop/ai practice/magento/e_com_data.json")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    product_texts = preprocess_product_data(products)
    embeddings = generate_embeddings(product_texts, model)
    index = create_faiss_index(embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    last_product = None
    cart = []
    conversation_history = []

    print("Hiii! I am your assistent (or 'exit' to quit):")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("\nThank you for shopping with us! Have a great day and happy shopping!\n")
            break

        # Add to cart
        if user_query.lower().startswith("add ") and "to cart" in user_query.lower():
            product_name = user_query[4:].lower().replace("to cart", "").strip()
            if product_name == "it" and last_product:
                cart.append(last_product)
                print(f"\nAdded '{last_product['name']}' to your cart.\n")
            else:
                matches = [p for p in products if p['name'].lower() == product_name]
                if matches:
                    cart.append(matches[0])
                    print(f"\nAdded '{matches[0]['name']}' to your cart.\n")
                else:
                    print("\nProduct not found.\n")
            continue

        # Show cart
        if user_query.lower() in ["show cart", "view cart", "cart"]:
            if not cart:
                print("\nYour cart is empty.\n")
            else:
                print("\nYour cart contains:")
                for idx, item in enumerate(cart, 1):
                    print(f"{idx}. {item['name']} - ${item['price']}")
                print()
            continue

        # Remove from cart
        if user_query.lower().startswith("remove ") and "from cart" in user_query.lower():
            product_name = user_query[7:].lower().replace("from cart", "").strip()
            for item in cart:
                if item['name'].lower() == product_name:
                    cart.remove(item)
                    print(f"\nRemoved '{item['name']}' from your cart.\n")
                    break
            else:
                print("\nProduct not found in cart.\n")
            continue

        # Checkout
        if user_query.lower() == "checkout":
            if not cart:
                print("\nYour cart is empty.\n")
            else:
                total = sum(item['price'] for item in cart)
                print("\nChecking out the following items:")
                for item in cart:
                    print(f"- {item['name']} - ${item['price']}")
                print(f"Total: ${total}\n")
                cart.clear()
            continue

        # Check if user is asking for a category
        user_query_lower = user_query.lower()
        found_category = None
        for p in products:
            cat = p.get('category', '').lower()
            if cat and cat in user_query_lower:
                found_category = cat
                break
        if found_category:
            # Return all products in that category in dictionary format
            category_products = [p for p in products if p.get('category', '').lower() == found_category]
            context = json.dumps(category_products, indent=2)
            print("\n" + generate_llm_response(user_query, context, llm) + "\n")
            if category_products:
                last_product = category_products[0]
            continue

        # Try to find exact product
        exact_matches = [p for p in products if user_query.lower() == p['name'].lower()]
        if exact_matches:
            last_product = exact_matches[0]
            context = format_product_details(last_product)
            recs = recommend_products(user_query, products, model, index, k=4)
            recs = [p for p in recs if p['name'].lower() != last_product['name'].lower()]
            if recs:
                context += "\n\nYou may also like:\n" + "\n\n".join([format_product_details(r) for r in recs[:3]])
            print("\n" + generate_llm_response(user_query, context, llm) + "\n")
            if recs:
                last_product = recs[0]
            continue

        # Otherwise, recommend products
        recs = recommend_products(user_query, products, model, index, k=3)
        context = "\n\n".join([format_product_details(prod) for prod in recs])
        print("\n" + generate_llm_response(user_query, context, llm) + "\n")
        if recs:
            last_product = recs[0]
