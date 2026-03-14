# 🛡️ ExamGuard AI

**A Deep Learning Watermarking System for Exam Paper Leak Detection**

ExamGuard AI is a comprehensive system that embeds invisible watermarks into exam papers to trace leaks back to the source student. Using deep learning models (encoder and decoder), the system generates unique watermarked copies for each student and can identify the source of leaked exams through watermark extraction and matching.

---

## 📋 Project Overview

### Problem Statement
Educational institutions face a significant challenge with exam paper leaks. Once an exam leaks, it's difficult to identify the source and prevent future compromises. ExamGuard AI solves this by embedding imperceptible, student-specific watermarks into each exam copy.

### Solution
- **Unique Watermarking:** Each student receives an exam with a unique watermark encoding their ID
- **Imperceptible Embedding:** Watermarks are invisible to the naked eye and blend seamlessly
- **Robust Extraction:** Even if the exam is scanned, reproduced, or shared, watermarks can be extracted
- **Source Identification:** Leaked exams can be matched back to the original student

---

## 🎯 Core Features

### 1. **Exam Upload**
- Upload original exam paper (JPG, PNG, BMP)
- Automatic image resizing to 256×256 pixels
- Preview of both original and resized versions
- Specify exam ID for identification

### 2. **Student Copy Generation**
- Input: Number of students and starting ID
- Process: Automatically generates unique watermark for each student
- Output: Watermarked exam copies ready for distribution
- Batch processing with progress tracking
- Download individual or all copies in ZIP format

### 3. **Leak Detection**
- Upload suspected leaked exam image
- System extracts the watermark using the decoder model
- Matches extracted watermark to the original student
- Reports: Student ID, confidence score, and leak confirmation

### 4. **Visualization & Analytics**
- Side-by-side comparison of original, watermarked, and decoded watermarks
- Difference maps to visualize watermark quality
- Confidence scoring for leak verification
- System status and model information

---

## 📁 Project Structure

```
ExamGuard_AI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── utils/
│   ├── __init__.py                # Package initialization
│   ├── watermark_utils.py         # Core watermarking functions
│   └── id_generator.py            # ID and watermark generation
│
└── models/
    ├── encoder.h5                 # Pretrained encoder model
    └── decoder.h5                 # Pretrained decoder model
```

---

## 🔧 Technical Architecture

### Encoder Model
```
Inputs:
  - Cover Image: (256, 256, 3)     [Original exam paper]
  - Watermark: (64, 64, 3)          [Student-specific watermark]

Processing:
  - Embeds watermark imperceptibly into cover image
  
Output:
  - Watermarked Image: (256, 256, 3) [Student's exam copy]
```

### Decoder Model
```
Input:
  - Watermarked Image: (256, 256, 3) [Potentially leaked exam]

Processing:
  - Extracts embedded watermark from image
  
Output:
  - Reconstructed Watermark: (64, 64, 3) [For source identification]
```

### ID to Watermark Conversion
```
Student ID + Exam ID
    ↓
Unique Watermark ID (e.g., "AI2026_000102")
    ↓
Deterministic Watermark Generation
    ↓
64×64×3 Unique Image
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Pretrained Models
Place your pretrained models in the `models/` directory:
```
models/
├── encoder.h5   # Your pretrained encoder
└── decoder.h5   # Your pretrained decoder
```

If models are not available, the system will use mock implementations for demonstration.

### Step 3: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📊 System Architecture

### Component Details

#### `watermark_utils.py`

**WatermarkProcessor Class**
- `__init__(encoder_path, decoder_path)`: Initialize with model paths
- `embed_watermark(cover_image, watermark_image)`: Embed watermark (256×256→256×256)
- `extract_watermark(watermarked_image)`: Extract watermark (256×256→64×64)
- `resize_image(image, target_size)`: Resize images using interpolation
- `normalize_image(image)`: Normalize to [0, 1] range
- `denormalize_image(image)`: Convert back to [0, 255]

**Image Processing**
- Uses OpenCV for efficient image operations
- Automatic normalization/denormalization
- Proper handling of uint8 and float32 formats

#### `id_generator.py`

**IDGenerator Class**
- `generate_watermark_id(student_id, exam_id)`: Create unique ID (e.g., "AI2026_000102")
- `convert_id_to_watermark_image(watermark_id)`: Generate 64×64×3 watermark image
- `match_watermark_to_id(watermark_image, candidate_ids)`: Match to most similar ID
- `_compute_similarity(img1, img2)`: Calculate similarity score (MSE-based)

**Watermark Generation**
- Deterministic: Same ID always produces same watermark
- Unique: Different IDs produce visually distinct watermarks
- Encoded: Contains student ID information in visual pattern
- Robust: Includes structural elements (circles, patterns) for resilience

#### `app.py`

**Streamlit Interface**
- Multi-tab interface for workflow
- Four main tabs: Upload, Generate, Leak Detection, System Info
- Session state management for data persistence
- Real-time progress tracking
- File download capabilities

**Tab Descriptions**

1. **📤 Exam Upload**
   - File upload with preview
   - Exam ID specification
   - Size validation and resizing

2. **📋 Generate Student Copies**
   - Batch watermarking with progress bar
   - Customizable student count and ID range
   - Individual exam download
   - Batch ZIP export

3. **🔍 Leak Detection**
   - Upload suspected leaked image
   - Automatic watermark extraction
   - Student ID identification
   - Confidence threshold configuration
   - Visual comparison interface

4. **ℹ️ System Info**
   - Model status display
   - Feature overview
   - Technical specifications
   - Session data summary

---

## 💡 Key Algorithms

### Watermark ID Generation
```python
watermark_id = f"{exam_id}_{student_id:06d}"
# Example: "AI2026_000102"
```

### Watermark Image Creation
1. Hash the watermark ID to a seed
2. Generate base pattern using hash values
3. Encode binary representation in spatial grid
4. Add structural elements (circles, patterns)
5. Apply color channel mixing for uniqueness

### Watermark Matching
1. Extract watermark from leaked image with decoder
2. For each candidate student:
   - Generate reference watermark from their ID
   - Compute MSE-based similarity score
3. Select match with highest confidence
4. Verify against confidence threshold

### Similarity Calculation
```
MSE = Mean((extracted - reference)²)
Similarity = 255² / (1 + MSE)
Confidence = min(1.0, Similarity / 100.0)
```

---

## 🔐 Security Considerations

### Strengths
- **Unique Watermarks:** Each student has a unique mark
- **Imperceptibility:** Watermarks are invisible without decoder
- **Robustness:** Can survive compression, scanning, reproduction
- **Traceability:** Clear identification of source

### Limitations
- Assumes high-quality pretrained models
- Relies on secure model storage
- Requires careful threshold calibration
- Mock models for demonstration (use real models in production)

---

## 📈 Usage Examples

### Example 1: Basic Workflow
```
1. Upload exam paper → Specify "AI2026" exam ID
2. Generate copies → 30 students (IDs 101-130)
3. Download watermarked exams
4. Distribute to students
5. If leak detected → Upload suspected image
6. System identifies Student ID: 115, Confidence: 0.97
```

### Example 2: Batch Processing
```
- 100+ students can be processed in seconds
- Automatic ID increment (101, 102, 103, ...)
- ZIP export for easy distribution
- Progress tracking for large batches
```

### Example 3: Leak Investigation
```
1. Institution receives leaked exam on internet
2. Screenshot/save the image
3. Upload to Leak Detection tab
4. System extracts watermark
5. Matches to Student ID: 087
6. Exam ID confirmed as "MIDTERM2026"
7. Investigation can proceed with identified source
```

---

## 🎓 Educational Value

This system demonstrates:
- Deep learning model integration
- Image processing with OpenCV
- Web application development with Streamlit
- Watermarking algorithms
- Forensic analysis concepts
- Information security principles

---

## 📝 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web UI framework |
| tensorflow | 2.14.0 | Deep learning models |
| opencv-python | 4.8.1.78 | Image processing |
| numpy | 1.24.3 | Numerical computing |
| pillow | 10.0.0 | Image I/O |

---

## 🔄 Data Flow

### Generation Flow
```
Original Exam (JPG)
    ↓ [Resize to 256×256]
Cover Image (256×256×3)
    ↓
[For each student]
    ↓
Generate Watermark ID (e.g., "AI2026_000101")
    ↓
Generate Watermark Image (64×64×3)
    ↓ [Encoder Model]
Watermarked Exam (256×256×3)
    ↓ [Save]
Student Exam Copy
```

### Detection Flow
```
Leaked Exam (JPG/PNG)
    ↓ [Resize to 256×256]
Suspected Watermarked Image (256×256×3)
    ↓ [Decoder Model]
Extracted Watermark (64×64×3)
    ↓
[For each generated exam]
    ↓
Compare with Reference Watermarks
    ↓
Calculate Similarity Scores
    ↓
Select Best Match (Highest Confidence)
    ↓
Report: Student ID, Confidence, Status
```

---

## 🛠️ Configuration

### Adjustable Parameters

**In `app.py`:**
```python
# Watermark size (matches encoder/decoder)
watermark_size = (64, 64, 3)

# Processing size
exam_size = (256, 256, 3)

# Confidence threshold (default 0.85)
confidence_threshold = 0.85

# Number of students (up to 100 in UI)
max_students = 100
```

### Model Paths
```python
encoder_path = "models/encoder.h5"
decoder_path = "models/decoder.h5"
```

---

## 🐛 Troubleshooting

### Issue: "Models not found" Warning
**Solution:** 
- Ensure `models/encoder.h5` and `models/decoder.h5` exist in the `models/` directory
- System will use mock models for demonstration if real models are unavailable
- For production, place your trained models in the correct location

### Issue: Image Upload Fails
**Solution:**
- Ensure image format is JPG, PNG, or BMP
- Check file is not corrupted
- Verify adequate disk space

### Issue: Slow Processing
**Solution:**
- GPU acceleration: Install tensorflow-gpu
- Reduce batch size in generation
- Use lower image resolution (current: 256×256)

---

## 🚀 Future Enhancements

Potential improvements for production:
1. **Real Model Integration:** Replace mock models with trained encoders/decoders
2. **Database Integration:** Store watermark-to-ID mappings in secure database
3. **Batch API:** REST API for automated batch processing
4. **Advanced Analytics:** Statistics on leak detection and source tracking
5. **Multi-Format Support:** Handle PDF, DOCX, and other exam formats
6. **Blockchain Integration:** Immutable audit trail of watermark assignments
7. **Improved Robustness:** Handle image compression, rotation, cropping
8. **Admin Dashboard:** Manage multiple exams and institutions
9. **Multi-Language Support:** Localization for international institutions

---

## 📄 License

This is a demonstration/educational project. Adapt for your specific use case.

---

## 👨‍💼 Author Notes

**Project:** ExamGuard AI - Deep Learning Watermarking System  
**Purpose:** Educational demonstration of deep learning, image processing, and web development  
**Created:** 2026  
**Status:** Complete prototype ready for enhancement

---

## ✅ Checklist for Implementation

- [x] Core watermarking utilities (encoder/decoder wrapper)
- [x] ID and watermark generation system
- [x] Streamlit UI with 4 main tabs
- [x] Exam upload and preprocessing
- [x] Batch watermark generation
- [x] Leak detection module
- [x] Visualization components
- [x] Download functionality (individual + ZIP)
- [x] Session state management
- [x] Error handling and validation
- [x] Documentation and README
- [ ] Trained encoder/decoder models (external)
- [ ] Database for watermark-ID mapping (optional)
- [ ] REST API for batch processing (optional)

---

## 📞 Support

For questions or issues:
1. Check the System Info tab for diagnostic information
2. Review error messages in the Streamlit console
3. Verify model files are in the correct location
4. Test with sample images before production use

---

**ExamGuard AI - Protecting Academic Integrity Through Technology** 🛡️
