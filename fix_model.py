import pickle

print("🔄 Loading model with NumPy 2.x...")
with open('MIPML.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Re-saving model with NumPy 1.x...")
with open('MIPML.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 Model re-saved successfully! Now it’s compatible with NumPy 1.x environments like Streamlit Cloud.")
