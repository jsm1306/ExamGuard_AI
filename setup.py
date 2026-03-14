"""
Quick setup and test script for ExamGuard AI
Run this to verify installation and create test data
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!\n")
    return True


def check_directories():
    """Verify directory structure."""
    print("📁 Checking directory structure...")
    
    required_dirs = ['utils', 'models']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            os.makedirs(dir_name, exist_ok=True)
    
    print("\n✅ Directory structure verified!\n")


def check_model_files():
    """Check if model files exist."""
    print("🤖 Checking model files...")
    
    models = [
        'models/encoder.h5',
        'models/decoder.h5'
    ]
    
    missing_models = []
    
    for model_path in models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✓ {model_path} ({size_mb:.2f} MB)")
        else:
            print(f"  ⚠ {model_path} - NOT FOUND (using mock for demo)")
            missing_models.append(model_path)
    
    if missing_models:
        print("\n📝 Note: Models not found. The system will use mock implementations.")
        print("To use real models, place them in the models/ directory:")
        for model in missing_models:
            print(f"      {model}")
        print("\nMock models are suitable for testing and demonstration.")
        return False
    
    print("\n✅ All models found!\n")
    return True


def create_sample_image(filename, width=256, height=256):
    """Create a sample image for testing."""
    print(f"Creating sample image: {filename}")
    
    # Create a gradient image with some structure
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient
    for i in range(height):
        image[i, :, 0] = int(255 * i / height)  # Red gradient
    for j in range(width):
        image[:, j, 1] = int(255 * j / width)   # Green gradient
    image[:, :, 2] = 128  # Blue constant
    
    # Add some text/pattern
    cv2.putText(image, 'EXAM', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255, 255, 255), 3)
    cv2.rectangle(image, (30, 30), (width-30, height-30), (255, 255, 255), 2)
    
    cv2.imwrite(filename, image)
    print(f"  ✓ Created: {filename}")


def create_sample_data():
    """Create sample images for testing."""
    print("\n🎨 Creating sample test images...")
    
    samples_dir = 'sample_images'
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create sample exam image
    sample_exam = os.path.join(samples_dir, 'sample_exam.png')
    if not os.path.exists(sample_exam):
        create_sample_image(sample_exam, 256, 256)
    
    print(f"\n✅ Sample images created in {samples_dir}/ directory\n")


def run_tests():
    """Run basic functionality tests."""
    print("🧪 Running basic tests...\n")
    
    try:
        from utils.watermark_utils import WatermarkProcessor
        from utils.id_generator import IDGenerator
        
        print("Testing ID Generator...")
        id_gen = IDGenerator()
        
        wm_id = id_gen.generate_watermark_id(101, "TEST2026")
        print(f"  ✓ Generated ID: {wm_id}")
        
        wm_image = id_gen.convert_id_to_watermark_image(wm_id)
        print(f"  ✓ Generated watermark shape: {wm_image.shape}")
        
        print("\n✅ All basic tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}\n")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n📚 Next Steps:\n")
    print("1. Review README.md for project overview")
    print("2. Check models/encoder.h5 and models/decoder.h5 exist")
    print("3. Run the Streamlit app:")
    print("   $ streamlit run app.py")
    print("\n4. Open browser to http://localhost:8501")
    print("5. Start with '📤 Exam Upload' tab")
    print("\n" + "=" * 60)


def main():
    """Run all setup checks."""
    print("\n" + "=" * 60)
    print("🛡️  ExamGuard AI - Setup Verification")
    print("=" * 60 + "\n")
    
    # Run checks
    deps_ok = check_dependencies()
    check_directories()
    models_ok = check_model_files()
    
    if deps_ok:
        create_sample_data()
        run_tests()
    
    print_next_steps()


if __name__ == "__main__":
    main()
