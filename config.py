"""
Configuration and constants for ExamGuard AI
"""

# ==================== IMAGE SIZES ====================
EXAM_IMAGE_SIZE = (256, 256, 3)        # Cover image size
WATERMARK_SIZE = (64, 64, 3)           # Watermark image size
EXAM_IMAGE_DIMS = (256, 256)           # For resize operations
WATERMARK_DIMS = (64, 64)              # For watermark generation

# ==================== PROCESSING ====================
IMAGE_NORMALIZATION_MAX = 255.0
CONFIDENCE_THRESHOLD_DEFAULT = 0.85
CONFIDENCE_THRESHOLD_MIN = 0.0
CONFIDENCE_THRESHOLD_MAX = 1.0

# ==================== MODEL PATHS ====================
ENCODER_MODEL_PATH = "models/encoder.h5"
DECODER_MODEL_PATH = "models/decoder.h5"

# ==================== UI LIMITS ====================
MAX_STUDENTS = 100
MIN_STUDENTS = 1
MAX_STUDENT_ID = 9999
MIN_STUDENT_ID = 1

# ==================== FILE UPLOAD ====================
ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
MAX_FILE_SIZE_MB = 200
UPLOAD_TIMEOUT_SECONDS = 30

# ==================== PROCESSING TIMEOUTS ====================
WATERMARKING_TIMEOUT_SECONDS = 60
EXTRACTION_TIMEOUT_SECONDS = 30
MATCHING_TIMEOUT_SECONDS = 60

# ==================== WATERMARK ID FORMAT ====================
WATERMARK_ID_FORMAT = "{exam_id}_{student_id:06d}"  # e.g., "AI2026_000101"

# ==================== SIMILARITY CALCULATIONS ====================
SIMILARITY_DENOMINATOR = 100.0          # Divisor for confidence calculation
MSE_BASE_VALUE = 255.0                  # Maximum pixel difference

# ==================== APP MESSAGES ====================
MESSAGES = {
    'setup_required': "Please upload an exam paper first in the 'Exam Upload' tab",
    'no_models': "Model files not found. Using mock models for demonstration.",
    'generation_success': "✓ Successfully generated watermarked exam copies",
    'leak_detected': "🚨 LEAK DETECTED",
    'no_leak_match': "⚠️ No matching watermark found with high confidence",
    'analysis_error': "Error analyzing image",
    'upload_error': "Error loading image",
}

# ==================== DEFAULTS ====================
DEFAULT_EXAM_ID = "AI2026"
DEFAULT_NUM_STUDENTS = 5
DEFAULT_STARTING_ID = 101

# ==================== VISUALIZATION ====================
IMAGE_DISPLAY_WIDTH = "100%"
HEATMAP_COLORMAP = 'hot'

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'

# ==================== EXPORT SETTINGS ====================
EXPORT_FORMAT = 'PNG'
ZIP_COMPRESSION = 'deflated'
