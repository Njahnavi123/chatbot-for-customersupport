import pickle

# Load trained model
with open("model/chatbot_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

print("Customer Support Chatbot (type 'exit' to quit)")

while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        print("Chatbot: Goodbye!")
        break

    input_vec = vectorizer.transform([user_input])
    response = model.predict(input_vec)
    print("Chatbot:", response[0])
