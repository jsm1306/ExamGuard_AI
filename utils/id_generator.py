"""
ID and watermark generation utilities for exam watermarking system.
"""

import numpy as np
import cv2
from typing import Tuple
import struct


class IDGenerator:
    """Generate unique IDs and convert to watermark images."""
    
    def __init__(self, watermark_size=(64, 64, 3)):
        """
        Initialize ID generator.
        
        Args:
            watermark_size (tuple): Size of watermark image (height, width, channels)
        """
        self.watermark_size = watermark_size
        self.id_map = {}  # Maps ID to watermark image for consistency
    
    def generate_watermark_id(self, student_id: int, exam_id: str) -> str:
        """
        Generate a unique watermark ID combining student and exam info.
        
        Args:
            student_id (int): Student ID number
            exam_id (str): Exam identifier (e.g., "AI2026")
        
        Returns:
            str: Unique watermark ID
        """
        # Create composite ID: exam_studentid format
        watermark_id = f"{exam_id}_{student_id:06d}"
        return watermark_id
    
    def convert_id_to_watermark_image(self, watermark_id: str) -> np.ndarray:
        """
        Convert watermark ID to a unique watermark image.
        
        Args:
            watermark_id (str): Watermark ID string
        
        Returns:
            np.ndarray: Watermark image of shape (64, 64, 3)
        """
        # Check if already generated
        if watermark_id in self.id_map:
            return self.id_map[watermark_id]
        
        # Create deterministic watermark from ID string
        # Use hash of ID as seed for reproducibility
        hash_seed = hash(watermark_id) % (2**31)
        np.random.seed(hash_seed)
        
        # Generate watermark as pattern based on ID
        watermark = self._generate_pattern_watermark(watermark_id)
        
        # Cache it
        self.id_map[watermark_id] = watermark.copy()
        
        return watermark
    
    def _generate_pattern_watermark(self, watermark_id: str) -> np.ndarray:
        """
        Generate a visually unique watermark pattern from ID string.
        
        Args:
            watermark_id (str): Watermark ID
        
        Returns:
            np.ndarray: Watermark image
        """
        h, w, c = self.watermark_size
        watermark = np.zeros((h, w, c), dtype=np.uint8)
        
        # Extract numeric values from ID for pattern generation
        id_bytes = watermark_id.encode()
        id_hash = sum([ord(x) * (i + 1) for i, x in enumerate(watermark_id)])
        
        # Create base pattern
        for i in range(h):
            for j in range(w):
                # Mix position, ID hash, and position-specific values
                pixel_val = (id_hash + i * w + j) % 256
                watermark[i, j, :] = pixel_val
        
        # Add color channels based on ID components
        for c_idx in range(c):
            channel_mod = (id_hash * (c_idx + 1)) % 256
            for i in range(h):
                for j in range(w):
                    val = (watermark[i, j, 0] + channel_mod) % 256
                    watermark[i, j, c_idx] = val
        
        # Draw binary pattern encoding the ID
        self._draw_id_pattern(watermark, watermark_id)
        
        # Add some visual structure
        self._add_structure(watermark, id_hash)
        
        return watermark
    
    def _draw_id_pattern(self, watermark: np.ndarray, watermark_id: str):
        """
        Draw an encoded pattern of the ID onto the watermark.
        
        Args:
            watermark (np.ndarray): Watermark to modify
            watermark_id (str): ID to encode
        """
        h, w, _ = watermark.shape
        
        # Encode ID as binary bits placed in a grid
        id_bytes = watermark_id.encode()
        bit_index = 0
        
        # Draw in 32x32 grid (bottom right quadrant)
        grid_size = 16
        cell_size = h // (2 * grid_size)
        
        for i in range(min(grid_size, len(id_bytes))):
            for j in range(8):  # 8 bits per byte
                if bit_index < len(id_bytes) * 8:
                    byte_val = id_bytes[bit_index // 8]
                    bit = (byte_val >> (bit_index % 8)) & 1
                    
                    # Draw bit at position
                    y_start = h // 2 + i * cell_size
                    x_start = w // 2 + (j * 4) * cell_size
                    
                    if y_start < h and x_start < w:
                        color = 255 if bit else 50
                        cv2.rectangle(watermark,
                                    (x_start, y_start),
                                    (min(x_start + cell_size, w), min(y_start + cell_size, h)),
                                    (color, color, color), -1)
                    
                    bit_index += 1
    
    def _add_structure(self, watermark: np.ndarray, seed: int):
        """
        Add structural elements to watermark for uniqueness.
        
        Args:
            watermark (np.ndarray): Watermark to modify
            seed (int): Seed for reproducibility
        """
        np.random.seed(seed)
        h, w, _ = watermark.shape
        
        # Add circular patterns
        for circle_idx in range(3):
            center_x = (seed + circle_idx * 100) % w
            center_y = (seed + circle_idx * 150) % h
            radius = 5 + (seed + circle_idx) % 10
            
            cv2.circle(watermark, (center_x, center_y), radius, 
                      (100, 100, 100), 1)
    
    def extract_id_from_watermark(self, watermark_image: np.ndarray) -> str:
        """
        Attempt to extract ID from watermark image.
        This is a simplified version - real implementation would use ML.
        
        Args:
            watermark_image (np.ndarray): Watermark image
        
        Returns:
            str: Extracted ID or "UNKNOWN"
        """
        # For now, return unknown as we'd need the actual mapping
        # In a real system, this would decode the pattern
        return "UNKNOWN"
    
    def match_watermark_to_id(self, watermark_image: np.ndarray, 
                            candidate_ids: list) -> Tuple[str, float]:
        """
        Match an extracted watermark to the most similar ID.
        
        Args:
            watermark_image (np.ndarray): Extracted watermark
            candidate_ids (list): List of candidate IDs to match against
        
        Returns:
            Tuple[str, float]: (matched_id, confidence_score)
        """
        if not candidate_ids:
            return "UNKNOWN", 0.0
        
        best_id = candidate_ids[0]
        best_score = 0.0
        
        for candidate_id in candidate_ids:
            # Generate reference watermark
            reference_watermark = self.convert_id_to_watermark_image(candidate_id)
            
            # Compute similarity (MSE-based)
            score = self._compute_similarity(watermark_image, reference_watermark)
            
            if score > best_score:
                best_score = score
                best_id = candidate_id
        
        # Convert score to confidence [0, 1]
        confidence = min(1.0, best_score / 100.0)
        
        return best_id, confidence
    
    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity between two images (inverse of MSE).
        
        Args:
            img1 (np.ndarray): First image
            img2 (np.ndarray): Second image
        
        Returns:
            float: Similarity score
        """
        # Ensure same shape
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Compute MSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Convert to similarity (higher is better)
        similarity = 255 ** 2 / (1 + mse)
        
        return similarity


def generate_watermark_id(student_id: int, exam_id: str) -> str:
    """
    Generate a unique watermark ID.
    
    Args:
        student_id (int): Student ID
        exam_id (str): Exam ID
    
    Returns:
        str: Unique watermark ID
    """
    generator = IDGenerator()
    return generator.generate_watermark_id(student_id, exam_id)


def convert_id_to_watermark_image(watermark_id: str) -> np.ndarray:
    """
    Convert ID to watermark image.
    
    Args:
        watermark_id (str): Watermark ID
    
    Returns:
        np.ndarray: Watermark image
    """
    generator = IDGenerator()
    return generator.convert_id_to_watermark_image(watermark_id)
