"""
Quick test of model loading and batch watermarking
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("ExamGuard AI - Batch Watermarking Setup Test")
print("=" * 70)

# Test 1: Import modules
print("\n[1/4] Testing imports...", end=" ")
try:
    from utils.watermark_utils import WatermarkProcessor
    from utils.id_generator import generate_watermark_id
    import cv2
    import numpy as np
    print("✓")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# Test 2: Load models
print("[2/4] Testing model loading...", end=" ")
try:
    processor = WatermarkProcessor()
    print("✓")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# Test 3: Generate watermark IDs
print("[3/4] Testing watermark ID generation...", end=" ")
try:
    test_ids = [generate_watermark_id(2310080001 + i, "AI2026") for i in range(3)]
    print(f"✓")
    print(f"      Sample IDs: {', '.join(test_ids)}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# Test 4: Test watermarking (with dummy images)
print("[4/4] Testing watermark embedding/extraction...", end=" ")
try:
    # Create dummy images
    cover = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    watermark = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # Try to embed
    watermarked = processor.embed_watermark(cover, watermark)
    
    # Try to extract
    extracted = processor.extract_watermark(watermarked)
    
    print("✓")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All tests passed! System is ready for batch generation.")
print("=" * 70)

print("\n📌 Next Steps:")
print("1. For IMAGE-based watermarking (PNG/JPG):")
print("   python batch_image_watermark.py")
print("\n2. For PDF-based watermarking (requires Poppler):")
print("   - Install Poppler: choco install poppler (admin mode)")
print("   - python batch_pdf_watermark.py")
print("\nNote: Image-based watermarking works without Poppler!")
print("=" * 70)
