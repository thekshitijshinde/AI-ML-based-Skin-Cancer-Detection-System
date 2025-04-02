import hashlib
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from web3 import Web3
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env file (keep keys secure)
load_dotenv()
INFURA_API_URL = f"https://sepolia.infura.io/v3/{os.getenv('INFURA_PROJECT_ID')}"
PRIVATE_KEY = os.getenv('PRIVATE_KEY')

# ✅ Connect to Ethereum blockchain
w3 = Web3(Web3.HTTPProvider(INFURA_API_URL))
wallet_address = w3.eth.account.from_key(PRIVATE_KEY).address
print(f"Connected to Ethereum. Wallet Address: {wallet_address}")

# ✅ Define class labels
classes = {
    0: "Actinic Keratoses (Cancer)",
    1: "Basal Cell Carcinoma (Cancer)",
    2: "Benign Keratosis (Non-Cancerous)",
    3: "Dermatofibroma (Non-Cancerous)",
    4: "Melanocytic Nevi (Non-Cancerous)",
    5: "Pyogenic Granulomas (Can lead to cancer)",
    6: "Melanoma (Cancer)"
}

# ✅ Load trained model
model = tf.keras.models.load_model("best_model.h5")

def predict_and_store(image_path):
    # ✅ Load and preprocess image
    img = image.load_img(image_path, target_size=(28, 28))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict skin cancer type
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[0][predicted_class])

    # ✅ Generate Hash for Blockchain Storage
    prediction_data = {
        "image_id": os.path.basename(image_path),
        "predicted_class": classes[predicted_class],
        "confidence": confidence
    }
    prediction_hash = hashlib.sha256(json.dumps(prediction_data).encode()).hexdigest()
    print(f"Prediction Hash: {prediction_hash}")

    # ✅ Blockchain Transaction - Storing Hash on Blockchain
    nonce = w3.eth.get_transaction_count(wallet_address)

    # 🔥 **Define gas_price before using it**
    gas_price = w3.eth.gas_price  # ✅ Fetch current gas price from Ethereum network

    tx = {
        'nonce': nonce,
        'to': wallet_address,  # Sending to self (to just store data)
        'value': 0,  # No ETH transfer
        'gas': 200000,
        'gasPrice': int(gas_price * 0.8),  # ✅ Now gas_price is defined
        'data': prediction_hash.encode().hex()  # Store hash in transaction data
    }

    # ✅ Sign & Send Transaction
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print(f"Transaction Sent! Hash: {tx_hash.hex()}")
    print("Prediction Stored on Blockchain Successfully! ✅")
    return tx_hash.hex()

# ✅ Run prediction + Blockchain storage whenever an image is uploaded
image_path = "tester.jpg"  # Change this to your uploaded image
tx_hash = predict_and_store(image_path)

# ✅ Output transaction hash for verification on Etherscan
print(f"View transaction on Etherscan: https://sepolia.etherscan.io/tx/{tx_hash}")
