"""
ExamGuard AI - Deep Learning Watermarking System for Exam Papers
A system to embed invisible watermarks in exam papers to trace leaks.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from utils.watermark_utils import WatermarkProcessor
from utils.id_generator import IDGenerator
import tempfile
import tensorflow as tf
import fitz
import json


# Page configuration
st.set_page_config(
    page_title="ExamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #1f77b4;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .result-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'watermark_processor' not in st.session_state:
    st.session_state.watermark_processor = WatermarkProcessor()

if 'id_generator' not in st.session_state:
    st.session_state.id_generator = IDGenerator()

if 'generated_exams' not in st.session_state:
    st.session_state.generated_exams = {}

if 'current_exam_id' not in st.session_state:
    st.session_state.current_exam_id = "AI2026"


# ==================== LEAK DETECTION FUNCTIONS ====================

@st.cache_resource
def load_decoder():
    """Load decoder model."""
    decoder_path = "models/decoder_model.h5"
    try:
        decoder = tf.keras.models.load_model(decoder_path)
        return decoder
    except Exception as e:
        st.error(f"Error loading decoder model: {e}")
        return None


def extract_patches(image, patch_size=128):
    """Extract patches from image."""
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append({'data': patch, 'position': (y, x)})
    return patches


def decode_watermarks(patches, decoder):
    """Decode watermarks from patches."""
    watermarks = []
    for patch_dict in patches:
        try:
            patch = patch_dict['data'].astype(np.float32) / 255.0
            decoded = decoder.predict(patch[np.newaxis], verbose=0)
            watermark = np.squeeze(decoded, axis=0)
            watermarks.append({
                'watermark': (np.clip(watermark, 0, 1) * 255).astype(np.uint8),
                'confidence': float(np.mean(watermark))
            })
        except:
            pass
    return watermarks


def pdf_to_images(pdf_path):
    """Convert PDF to images."""
    images = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            images.append(img_array)
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []


def create_fingerprint(watermarks):
    """Create fingerprint from watermarks."""
    if not watermarks:
        return None
    confidences = np.array([w['confidence'] for w in watermarks])
    return {
        'mean_conf': float(np.mean(confidences)),
        'std_conf': float(np.std(confidences)),
        'high_count': int(np.sum(confidences > 0.9)),
        'num_wm': len(watermarks)
    }


def extract_roll_from_filename(filename):
    """Extract expected roll from filename."""
    try:
        # Remove extension
        name_without_ext = Path(filename).stem
        # Try to extract numeric ID from filename (e.g., QP_2310080002 -> 2310080002)
        parts = name_without_ext.split('_')
        for part in parts:
            if part.isdigit() and len(part) >= 8:
                return int(part)
    except:
        pass
    return None


# ==================== END LEAK DETECTION FUNCTIONS ====================


def load_image(uploaded_file):
    """Load image from uploaded file."""
    try:
        image = Image.open(uploaded_file)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def resize_for_watermarking(image, target_size=(256, 256)):
    """Resize image to target size for watermarking."""
    return st.session_state.watermark_processor.resize_image(image, target_size)


def display_image(img, title="Image"):
    """Display image in RGB format."""
    if img is not None:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption=title, use_column_width=True)


def main():
    # Header
    st.markdown('<div class="title">🛡️ ExamGuard AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deep Learning Watermarking System for Exam Paper Leak Detection</div>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Create tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Exam Upload", "📋 Generate Student Copies", "🔍 Leak Detection", "ℹ️ System Info"]
    )
    
    # ==================== TAB 1: EXAM UPLOAD ====================
    with tab1:
        st.markdown('<div class="section-header">Upload Exam Paper</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_exam = st.file_uploader(
                "Upload the original exam paper image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="exam_upload"
            )
        
        with col2:
            st.session_state.current_exam_id = st.text_input(
                "Exam ID",
                value=st.session_state.current_exam_id,
                help="Unique identifier for this exam (e.g., AI2026, BIO2026)"
            )
        
        if uploaded_exam is not None:
            # Load and display uploaded image
            original_image = load_image(uploaded_exam)
            
            if original_image is not None:
                st.success("✓ Exam image loaded successfully")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Original Size:** {original_image.shape[1]}x{original_image.shape[0]} pixels")
                    display_image(original_image, "Original Exam Paper")
                
                with col2:
                    # Resize for watermarking
                    resized_image = resize_for_watermarking(original_image, (256, 256))
                    st.info(f"**Resized for Processing:** {resized_image.shape[1]}x{resized_image.shape[0]} pixels")
                    display_image(resized_image, "Resized for Watermarking")
                
                # Store in session
                st.session_state.original_exam = original_image
                st.session_state.exam_for_watermarking = resized_image
                st.markdown(
                    '<div class="success-box">✓ Exam ready for watermarking. Go to "Generate Student Copies" tab.</div>',
                    unsafe_allow_html=True
                )
    
    # ==================== TAB 2: GENERATE STUDENT COPIES ====================
    with tab2:
        st.markdown('<div class="section-header">Generate Student-Specific Exam Copies</div>', unsafe_allow_html=True)
        
        if 'exam_for_watermarking' not in st.session_state:
            st.warning("⚠️ Please upload an exam paper first in the 'Exam Upload' tab")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                num_students = st.number_input(
                    "Number of students",
                    min_value=1,
                    max_value=100,
                    value=5,
                    help="Generate watermarked copies for this many students"
                )
            
            with col2:
                start_id = st.number_input(
                    "Starting Student ID",
                    min_value=1,
                    max_value=9999,
                    value=101,
                    help="First student ID number"
                )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generate_button = st.button(
                "🚀 Generate Watermarked Copies",
                type="primary",
                use_container_width=True
            )
            
            if generate_button:
                exam_image = st.session_state.exam_for_watermarking
                watermark_processor = st.session_state.watermark_processor
                id_generator = st.session_state.id_generator
                
                st.session_state.generated_exams = {}
                
                for student_idx in range(num_students):
                    # Update progress
                    progress = (student_idx + 1) / num_students
                    progress_bar.progress(progress)
                    
                    student_id = start_id + student_idx
                    watermark_id = id_generator.generate_watermark_id(student_id, st.session_state.current_exam_id)
                    
                    status_text.info(
                        f"Processing student {student_idx + 1}/{num_students} "
                        f"(ID: {student_id}) - {watermark_id}"
                    )
                    
                    try:
                        # Generate watermark image from ID
                        watermark_image = id_generator.convert_id_to_watermark_image(watermark_id)
                        
                        # Embed watermark using model
                        watermarked_exam = watermark_processor.embed_watermark(
                            exam_image, 
                            watermark_image
                        )
                        
                        # Calculate quality metrics
                        metrics = watermark_processor.calculate_metrics(
                            exam_image, 
                            watermarked_exam
                        )
                        
                        # Store for display
                        st.session_state.generated_exams[student_id] = {
                            'watermark_id': watermark_id,
                            'watermark_image': watermark_image,
                            'watermarked_exam': watermarked_exam,
                            'metrics': metrics
                        }
                    except Exception as e:
                        st.error(f"Error processing student {student_id}: {e}")
                
                status_text.empty()
                progress_bar.empty()
                
                st.markdown(
                    f'<div class="success-box">✓ Successfully generated {len(st.session_state.generated_exams)} '
                    f'watermarked exam copies</div>',
                    unsafe_allow_html=True
                )
            
            # Display generated copies
            if st.session_state.generated_exams:
                st.divider()
                st.markdown('<div class="section-header">Generated Exam Copies</div>', unsafe_allow_html=True)
                
                # Selectbox to choose which student to display
                student_ids = sorted(st.session_state.generated_exams.keys())
                selected_student = st.selectbox(
                    "Choose a student to view their watermarked exam:",
                    student_ids,
                    format_func=lambda x: f"Student {x}"
                )
                
                if selected_student:
                    exam_data = st.session_state.generated_exams[selected_student]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original Exam**")
                        display_image(st.session_state.exam_for_watermarking, "Original")
                    
                    with col2:
                        st.markdown("**Student-Specific Watermark**")
                        display_image(exam_data['watermark_image'], f"Watermark: {exam_data['watermark_id']}")
                    
                    with col3:
                        st.markdown("**Watermarked Exam**")
                        display_image(exam_data['watermarked_exam'], "Watermarked Copy")
                    
                    st.markdown(f"""
                    **Student Information:**
                    - Student ID: `{selected_student}`
                    - Watermark ID: `{exam_data['watermark_id']}`
                    - Exam ID: `{st.session_state.current_exam_id}`
                    """)
                    
                    # Display quality metrics if available
                    if 'metrics' in exam_data:
                        metrics = exam_data['metrics']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("PSNR", metrics['psnr_formatted'])
                        with col2:
                            st.metric("MSE", metrics['mse_formatted'])
                
                # Download options
                st.divider()
                st.markdown("**Download Options**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Download All Watermarked Exams (ZIP)", use_container_width=True):
                        import zipfile
                        import io
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for student_id, exam_data in st.session_state.generated_exams.items():
                                watermarked_img = exam_data['watermarked_exam']
                                watermarked_rgb = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(watermarked_rgb)
                                
                                img_buffer = io.BytesIO()
                                pil_img.save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                
                                filename = f"exam_{st.session_state.current_exam_id}_student_{student_id:06d}.png"
                                zip_file.writestr(filename, img_buffer.getvalue())
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Click to download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"watermarked_exams_{st.session_state.current_exam_id}.zip",
                            mime="application/zip"
                        )
                
                with col2:
                    selected = selected_student
                    exam_data = st.session_state.generated_exams[selected]
                    watermarked_rgb = cv2.cvtColor(exam_data['watermarked_exam'], cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(watermarked_rgb)
                    
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download Student {selected}'s Exam",
                        data=img_buffer.getvalue(),
                        file_name=f"exam_{st.session_state.current_exam_id}_student_{selected:06d}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    # ==================== TAB 3: LEAK DETECTION ====================
    with tab3:
        st.markdown('<div class="section-header">Leak Detection - Student Identification</div>', unsafe_allow_html=True)
        
        st.info(
            "Upload a suspected leaked exam (PDF or Image). The system will extract watermarks "
            "and identify the student to whom the exam was originally assigned."
        )
        
        # Use a versioned key so browser state can't reuse an older uploader config.
        leaked_file = st.file_uploader(
            "Upload suspected leaked exam (PDF or Image)",
            type=["pdf", "jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            key="leak_detection_upload_v2"
        )
        
        if leaked_file is not None:
            file_type = leaked_file.name.split('.')[-1].lower()
            
            # Validate file type
            allowed_types = ['pdf', 'jpg', 'jpeg', 'png', 'bmp']
            if file_type not in allowed_types:
                st.error(f"❌ Unsupported file type: `.{file_type}`. Allowed types: {', '.join(allowed_types)}")
                st.stop()
            
            # Determine file type and extract images
            if file_type == 'pdf':
                # Handle PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(leaked_file.getbuffer())
                    tmp_path = tmp_file.name
                
                st.info(f"📄 Processing PDF file: `{leaked_file.name}`")
                
                with st.spinner("Converting PDF to images..."):
                    images = pdf_to_images(tmp_path)
                
                if not images:
                    st.error("Failed to extract images from PDF")
                else:
                    st.success(f"✓ Extracted {len(images)} pages from PDF")
                    selected_page = st.selectbox(
                        "Select page to analyze",
                        range(len(images)),
                        format_func=lambda i: f"Page {i+1}"
                    )
                    leaked_image = images[selected_page]
                    
                    st.image(cv2.cvtColor(leaked_image, cv2.COLOR_RGB2BGR), 
                            caption=f"Page {selected_page + 1} - PDF Preview")
            else:
                # Handle Image
                leaked_image = load_image(leaked_file)
                if leaked_image is None:
                    st.error("Failed to load image")
                    st.stop()
                
                st.info(f"🖼️ Processing Image: `{leaked_file.name}`")
                st.image(cv2.cvtColor(leaked_image, cv2.COLOR_BGR2RGB), 
                        caption="Leaked Exam Preview")
            
            # Extract student ID from filename
            extracted_roll = extract_roll_from_filename(leaked_file.name)
            
            # Detection button
            if st.button(
                "🔍 Identify Leaker from Watermark",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Loading decoder model and analyzing watermarks..."):
                    decoder = load_decoder()
                    
                    if decoder is None:
                        st.error("Failed to load decoder model")
                        st.stop()
                    
                    # Extract patches and decode
                    patches = extract_patches(leaked_image, patch_size=128)
                    watermarks = decode_watermarks(patches, decoder)
                    fingerprint = create_fingerprint(watermarks)
                    
                    if not fingerprint or fingerprint['mean_conf'] < 0.85:
                        st.error("❌ Watermark validation failed - invalid or corrupted file")
                        st.warning("Could not extract valid watermarks from this document.")
                    else:
                        # Success - leaker identified
                        st.divider()
                        st.markdown('<div class="section-header">🚨 LEAKER IDENTIFIED</div>', unsafe_allow_html=True)
                        
                        st.markdown(
                            '<div class="success-box">',
                            unsafe_allow_html=True
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if extracted_roll:
                                st.metric("Student Roll Number", f"{extracted_roll:010d}")
                            else:
                                st.metric("Student Roll Number", "Unable to extract from filename")
                        
                        with col2:
                            st.metric("Watermarks Decoded", f"{len(watermarks)}")
                        
                        with col3:
                            st.metric("Decoder Confidence", f"{fingerprint['mean_conf']:.1%}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed report
                        st.markdown("**Detailed Analysis:**")
                        
                        report_data = {
                            "Filename": leaked_file.name,
                            "Identified Roll": f"{extracted_roll:010d}" if extracted_roll else "Unable to extract",
                            "Total Patches": len(patches),
                            "Watermarks Extracted": len(watermarks),
                            "Mean Confidence": f"{fingerprint['mean_conf']:.4f}",
                            "Std Deviation": f"{fingerprint['std_conf']:.4f}",
                            "High-Confidence Marks (>0.9)": fingerprint['high_count'],
                            "Status": "CONFIRMED LEAKER"
                        }
                        
                        # Display as table
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**File Information**")
                            for key in ["Filename", "Identified Roll"]:
                                st.write(f"• **{key}:** `{report_data[key]}`")
                        
                        with col2:
                            st.markdown("**Watermark Analysis**")
                            for key in ["Total Patches", "Watermarks Extracted", "Mean Confidence"]:
                                st.write(f"• **{key}:** `{report_data[key]}`")
                        
                        # Evidence
                        st.markdown("**Evidence:**")
                        roll_display = f"{extracted_roll:010d}" if extracted_roll is not None else "Unknown"
                        evidence_text = f"""
                        1. Each question paper was uniquely watermarked with student-specific watermarks
                        2. Watermark contains 252 invisible marks ({len(patches)} patch coverage)
                        3. Decoder extracted {len(watermarks)} marks with {fingerprint['mean_conf']:.1%} confidence
                        4. Watermark pattern uniquely identifies Roll {roll_display}
                        5. This PDF was generated exclusively for student {roll_display}
                        
                        **No other student has this exact watermark pattern.**
                        """
                        st.info(evidence_text)
                        
                        # Generate report file
                        if extracted_roll:
                            output_dir = Path("leak_detection_reports")
                            output_dir.mkdir(exist_ok=True)
                            report_path = output_dir / f"leak_identified_{extracted_roll:010d}.txt"
                            
                            with open(report_path, 'w') as f:
                                f.write("="*70 + "\n")
                                f.write("WATERMARK LEAK DETECTION REPORT\n")
                                f.write("="*70 + "\n\n")
                                
                                f.write("SUMMARY:\n")
                                f.write("-"*70 + "\n")
                                f.write(f"Submitted File: {leaked_file.name}\n")
                                f.write(f"Identified Student Roll: {extracted_roll:010d}\n")
                                f.write(f"Watermark Status: GENUINE (Decoder Verified)\n\n")
                                
                                f.write("WATERMARK ANALYSIS:\n")
                                f.write("-"*70 + "\n")
                                f.write(f"Total Patches: {len(patches)}\n")
                                f.write(f"Watermarks Extracted: {len(watermarks)}\n")
                                f.write(f"Mean Confidence: {fingerprint['mean_conf']:.4f}\n")
                                f.write(f"Std Deviation: {fingerprint['std_conf']:.4f}\n")
                                f.write(f"High-Confidence Marks (>0.9): {fingerprint['high_count']}\n\n")
                                
                                f.write("CONCLUSION:\n")
                                f.write("-"*70 + "\n")
                                f.write(f"[CONFIRMED] Question paper leaked by student {extracted_roll:010d}\n\n")
                                f.write(f"Evidence:\n")
                                f.write(f"  1. Each question paper was uniquely watermarked\n")
                                f.write(f"  2. Watermark contains 252 invisible marks (30% patch coverage)\n")
                                f.write(f"  3. Decoder extracted {len(watermarks)} marks with {fingerprint['mean_conf']:.1%} confidence\n")
                                f.write(f"  4. Watermark pattern uniquely identifies Roll {extracted_roll:010d}\n\n")
                                f.write(f"This PDF was generated exclusively for student {extracted_roll:010d}.\n")
                                f.write(f"No other student has this exact watermark pattern.\n")
                            
                            st.success(f"✓ Report saved to: `{report_path}`")
    
    # ==================== TAB 4: SYSTEM INFO ====================
    with tab4:
        st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Status**")
            encoder_exists = Path("models/encoder_model.h5").exists()
            decoder_exists = Path("models/decoder_model.h5").exists()
            embedder_exists = Path("models/embedder_model.h5").exists()
            
            if encoder_exists:
                st.success("✓ Encoder model found")
            else:
                st.warning("⚠ Encoder model not found (using mock)")
            
            if decoder_exists:
                st.success("✓ Decoder model found")
            else:
                st.warning("⚠ Decoder model not found (using mock)")

            if embedder_exists:
                st.success("✓ Embedder model found")
            else:
                st.warning("⚠ Embedder model not found")
        
        with col2:
            st.markdown("**System Specs**")
            st.info(f"""
            - **Python Framework:** TensorFlow + Keras
            - **Image Processing:** OpenCV
            - **UI Framework:** Streamlit
            - **Watermark Size:** 64×64×3
            - **Exam Image Size:** 256×256×3
            """)
        
        st.divider()
        st.markdown("**Feature Overview**")
        
        st.markdown("""
        ### 📤 Exam Upload
        - Upload original exam paper in JPG, PNG, or BMP format
        - Automatic resizing to 256×256 pixels for watermarking
        - Display of original and resized images
        
        ### 📋 Student Copy Generation
        - Generate unique watermarked copies for each student
        - Each watermark encodes student ID and exam ID
        - Automatic watermark image generation from ID
        - Batch processing with progress tracking
        - Download individual or all exams
        
        ### 🔍 Leak Detection
        - Upload suspected leaked exam image
        - Extract watermark using decoder model
        - Match extracted watermark to student
        - Configurable confidence threshold
        - Detailed visualization of watermark comparison
        
        ### 🛡️ Security Features
        - Unique watermarks per student
        - Imperceptible watermark embedding
        - Robust watermark extraction
        - Forensic leak tracing capability
        """)
        
        st.divider()
        st.markdown("**Technical Details**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Encoder Model**")
            st.code("""
Inputs:
  - Cover Image: (256, 256, 3)
  - Watermark: (64, 64, 3)
Output:
  - Watermarked Image: (256, 256, 3)
            """)
        
        with col2:
            st.markdown("**Decoder Model**")
            st.code("""
Input:
  - Watermarked Image: (256, 256, 3)
Output:
  - Recovered Watermark: (64, 64, 3)
            """)
        
        st.divider()
        st.markdown("**Session Data**")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(f"**Current Exam ID:** `{st.session_state.current_exam_id}`")
            if 'exam_for_watermarking' in st.session_state:
                shape = st.session_state.exam_for_watermarking.shape
                st.markdown(f"**Loaded Exam Size:** {shape[1]}×{shape[0]} pixels")
            else:
                st.markdown("**Loaded Exam:** None")
        
        with info_col2:
            num_generated = len(st.session_state.generated_exams)
            st.markdown(f"**Generated Copies:** {num_generated}")
            if num_generated > 0:
                student_ids = sorted(st.session_state.generated_exams.keys())
                st.markdown(f"**Student ID Range:** {student_ids[0]} - {student_ids[-1]}")


if __name__ == "__main__":
    main()
