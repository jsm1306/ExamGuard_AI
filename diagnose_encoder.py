"""Diagnose encoder model structure"""
import tensorflow as tf
import numpy as np

print("="*70)
print("ENCODER MODEL DIAGNOSTIC")
print("="*70)

encoder = tf.keras.models.load_model('models/encoder_model.h5')

print(f"\nNumber of inputs: {len(encoder.inputs)}")
for i, inp in enumerate(encoder.inputs):
    print(f"  Input {i}: shape={inp.shape}, dtype={inp.dtype}")

print(f"\nNumber of outputs: {len(encoder.outputs)}")  
for i, out in enumerate(encoder.outputs):
    print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")

print(f"\nModel summary:")
encoder.summary()

# Test with different input shapes
print("\n" + "="*70)
print("TESTING DIFFERENT INPUT COMBINATIONS")
print("="*70)

# Try single input (watermark)
try:
    print("\n1. Single input (1, 32, 32, 3):")
    dummy = np.random.randn(1, 32, 32, 3).astype(np.float32)
    out = encoder.predict(dummy, verbose=0)
    print(f"   Output shape: {out.shape}")
except Exception as e:
    print(f"   [ERROR] {str(e)[:80]}")

# Try two inputs [patch, watermark]
try:
    print("\n2. Two inputs: [patch (1,256,256,3), watermark (1,32,32,3)]:")
    patch = np.random.randn(1, 256, 256, 3).astype(np.float32)
    wm = np.random.randn(1, 32, 32, 3).astype(np.float32)
    out = encoder.predict([patch, wm], verbose=0)
    print(f"   Output shape: {out.shape}")
except Exception as e:
    print(f"   [ERROR] {str(e)[:80]}")

# Try two inputs with different shapes
try:
    print("\n3. Two inputs: [patch (1,128,128,3), watermark (1,32,32,3)]:")
    patch = np.random.randn(1, 128, 128, 3).astype(np.float32)
    wm = np.random.randn(1, 32, 32, 3).astype(np.float32)
    out = encoder.predict([patch, wm], verbose=0)
    print(f"   Output shape: {out.shape}")
except Exception as e:
    print(f"   [ERROR] {str(e)[:80]}")

print("\n" + "="*70)
