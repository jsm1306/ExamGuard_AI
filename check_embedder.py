"""Check embedder model inputs"""
import tensorflow as tf

emb = tf.keras.models.load_model('models/embedder_model.h5')

print(f"Embedder inputs: {len(emb.inputs)}")
for i, inp in enumerate(emb.inputs):
    print(f"  Input {i}: {inp.shape}")

print(f"\nEmbedder output: {emb.output.shape}")

print("\nEmbedder model:")
emb.summary()
