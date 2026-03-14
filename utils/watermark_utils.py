"""
Watermark utilities for embedding and extracting watermarks using TensorFlow models.
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path


class WatermarkProcessor:
    """Handles watermark embedding using patch-based approach with encoder model."""
    
    def __init__(self, encoder_path="models/encoder_model.h5"):
        """
        Initialize the watermark processor with encoder model.
        
        Args:
            encoder_path (str): Path to the encoder model for patch watermarking
        """
        self.encoder = None
        # Resolve path relative to this file's location, not current working directory
        if not Path(encoder_path).is_absolute():
            encoder_path = Path(__file__).parent.parent / encoder_path
        self.encoder_path = encoder_path
        self._load_encoder()
    
    def _load_encoder(self):
        """Load the encoder model for patch watermarking."""
        print("Loading encoder model for patch watermarking...", end=" ", flush=True)
        try:
            if Path(self.encoder_path).exists():
                self.encoder = tf.keras.models.load_model(self.encoder_path)
                print("[OK] Encoder model loaded")
            else:
                print("[NOT FOUND]")
                print(f"  Encoder model not found at {self.encoder_path}")
                print("  Download or train encoder_model.h5")
                self.encoder = None
        except Exception as e:
            print(f"[ERROR]")
            print(f"  Error loading encoder: {str(e)[:80]}")
            self.encoder = None
    
    def _build_encoder(self):
        """Build encoder component"""
        from tensorflow.keras import layers, Model
        inputs = layers.Input(shape=(32, 32, 3), name='watermark_input')
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(24, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(24, 1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Conv2D(3, 1, padding='same', activation='tanh')(x)
        return Model(inputs, encoded, name='Encoder')
    
    def _build_embedder(self):
        """Build embedder component"""
        from tensorflow.keras import layers, Model
        cover = layers.Input(shape=(128, 128, 3), name='cover_input')
        watermark = layers.Input(shape=(32, 32, 3), name='watermark_input')
        
        w_up = layers.UpSampling2D(size=(4, 4))(watermark)
        merged = layers.Concatenate()([cover, w_up])
        
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(merged)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        watermarked = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
        
        return Model([cover, watermark], watermarked, name='Embedder')
    
    def _build_decoder(self):
        """Build decoder component"""
        from tensorflow.keras import layers, Model
        inputs = layers.Input(shape=(128, 128, 3), name='watermarked_input')
        
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
        
        # Extract watermark region (center 32x32)
        center_crop = layers.Cropping2D(cropping=((48, 48), (48, 48)))(x)
        decoded = layers.Conv2D(3, 1, padding='same', activation='sigmoid')(center_crop)
        
        return Model(inputs, decoded, name='Decoder')
    
    def embed_watermark(self, cover_image, watermark_image):
        """
        Embed watermark into cover image using patch-based approach.
        
        Strategy for invisibility:
        1. Split image into 128×128 patches
        2. Watermark only random 30% of patches (avoid visible patterns)
        3. Scale perturbation strength to 0.02 (invisible to human eye)
        4. Reconstruct full page
        
        This ensures watermark is:
        - Invisible to humans
        - Robust for decoder model
        - Doesn't create visible patterns
        
        Args:
            cover_image (np.ndarray): Cover image (H, W, 3), dtype uint8
            watermark_image (np.ndarray): Watermark image (32, 32, 3), dtype uint8
        
        Returns:
            np.ndarray: Watermarked image same shape as input, dtype uint8
        """
        try:
            h, w = cover_image.shape[:2]
            patch_size = 128
            
            # Normalize inputs to [0, 1]
            cover_normalized = cover_image.astype(np.float32) / 255.0
            watermark_normalized = watermark_image.astype(np.float32) / 255.0
            
            # Result image
            result = np.copy(cover_normalized)
            
            if self.encoder is None:
                print("[!] Encoder not available, using mock watermarking")
                return self._mock_embed_patches(cover_image, watermark_image)
            
            # Collect all valid patch positions
            patch_positions = []
            for y in range(0, h - patch_size, patch_size):
                for x in range(0, w - patch_size, patch_size):
                    patch_positions.append((y, x))
            
            # Randomly select 30% of patches to watermark
            num_patches_to_watermark = max(1, int(len(patch_positions) * 0.3))
            selected_indices = np.random.choice(
                len(patch_positions), 
                size=num_patches_to_watermark, 
                replace=False
            )
            
            patches_processed = 0
            
            # Process only selected patches
            for idx in selected_indices:
                y, x = patch_positions[idx]
                
                # Extract patch
                patch = cover_normalized[y:y + patch_size, x:x + patch_size]
                
                # Try embedding with encoder
                try:
                    # Add batch dimension
                    patch_batch = patch[np.newaxis]  # (1, 128, 128, 3)
                    watermark_batch = watermark_normalized[np.newaxis]  # (1, 32, 32, 3)
                    
                    # Attempt encoder prediction with both inputs
                    try:
                        watermarked_batch = self.encoder.predict(
                            [patch_batch, watermark_batch],
                            verbose=0
                        )
                    except (TypeError, ValueError):
                        # If encoder only takes one input, use fallback
                        watermarked_batch = self._embed_patch_fallback(
                            patch_batch, watermark_batch
                        )
                    
                    # Handle different output formats
                    if isinstance(watermarked_batch, (list, tuple)):
                        watermarked_patch = np.squeeze(watermarked_batch[0], axis=0)
                    else:
                        watermarked_patch = np.squeeze(watermarked_batch, axis=0)
                    
                    # SCALE DOWN PERTURBATION: Only use 0.02 (2%) of the watermark signal
                    # This makes it imperceptible to humans
                    perturbation_strength = 0.02
                    blended = (1.0 - perturbation_strength) * patch + perturbation_strength * watermarked_patch
                    
                    result[y:y + patch_size, x:x + patch_size] = blended
                    patches_processed += 1
                    
                except Exception as e:
                    # If encoder fails, keep original patch
                    pass
            
            # Handle remaining edges if image dimensions aren't multiples of 128
            # Right edge (only partial watermarking)
            if w % patch_size != 0:
                x = (w // patch_size) * patch_size
                if x < w:
                    for y in range(0, h - patch_size, patch_size):
                        # Only watermark edge if randomly selected
                        if np.random.random() < 0.3:
                            edge_patch = cover_normalized[y:y + patch_size, x:w]
                            padded = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                            padded[:, :edge_patch.shape[1]] = edge_patch
                            
                            patch_batch = padded[np.newaxis]
                            watermark_batch = watermark_normalized[np.newaxis]
                            
                            try:
                                watermarked_batch = self.encoder.predict(
                                    [patch_batch, watermark_batch],
                                    verbose=0
                                )
                                if isinstance(watermarked_batch, (list, tuple)):
                                    watermarked_patch = np.squeeze(watermarked_batch[0], axis=0)
                                else:
                                    watermarked_patch = np.squeeze(watermarked_batch, axis=0)
                                
                                perturbation_strength = 0.02
                                blended = (1.0 - perturbation_strength) * padded + perturbation_strength * watermarked_patch
                                result[y:y + patch_size, x:w] = blended[:, :edge_patch.shape[1]]
                            except:
                                pass
            
            # Bottom edge (only partial watermarking)
            if h % patch_size != 0:
                y = (h // patch_size) * patch_size
                if y < h:
                    for x in range(0, w - patch_size, patch_size):
                        # Only watermark edge if randomly selected
                        if np.random.random() < 0.3:
                            edge_patch = cover_normalized[y:h, x:x + patch_size]
                            padded = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                            padded[:edge_patch.shape[0], :] = edge_patch
                            
                            patch_batch = padded[np.newaxis]
                            watermark_batch = watermark_normalized[np.newaxis]
                            
                            try:
                                watermarked_batch = self.encoder.predict(
                                    [patch_batch, watermark_batch],
                                    verbose=0
                                )
                                if isinstance(watermarked_batch, (list, tuple)):
                                    watermarked_patch = np.squeeze(watermarked_batch[0], axis=0)
                                else:
                                    watermarked_patch = np.squeeze(watermarked_batch, axis=0)
                                
                                perturbation_strength = 0.02
                                blended = (1.0 - perturbation_strength) * padded + perturbation_strength * watermarked_patch
                                result[y:h, x:x + patch_size] = blended[:edge_patch.shape[0], :]
                            except:
                                pass
            
            # Clip to [0, 1] and denormalize
            result = np.clip(result, 0, 1)
            result = (result * 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] During patch watermarking: {e}")
            raise
    
    def _embed_patch_fallback(self, patch_batch, watermark_batch):
        """
        Fallback method when encoder doesn't support both inputs.
        
        Scales perturbation to 0.02 for invisibility.
        """
        # Simple watermark blending on patch with small perturbation
        patch = np.squeeze(patch_batch, axis=0)
        watermark = np.squeeze(watermark_batch, axis=0)
        watermark_expanded = cv2.resize(watermark, (128, 128))
        
        # Blend with 0.02 perturbation strength (invisible to humans)
        perturbation_strength = 0.02
        blended = (1.0 - perturbation_strength) * patch + perturbation_strength * watermark_expanded
        return np.clip(blended, 0, 1)[np.newaxis]
    
    def _mock_embed_patches(self, cover_image, watermark_image):
        """Mock watermark embedding with 30% patches and 0.02 perturbation."""
        result = cover_image.copy().astype(np.float32) / 255.0
        h, w = cover_image.shape[:2]
        patch_size = 128
        
        # Expand watermark to patch size
        watermark_expanded = cv2.resize(watermark_image, (patch_size, patch_size))
        watermark_normalized = watermark_expanded.astype(np.float32) / 255.0
        
        # Collect all valid patch positions
        patch_positions = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch_positions.append((y, x))
        
        # Randomly select 30% of patches
        num_patches_to_watermark = max(1, int(len(patch_positions) * 0.3))
        selected_indices = np.random.choice(
            len(patch_positions),
            size=num_patches_to_watermark,
            replace=False
        )
        
        # Apply watermark with small perturbation (0.02)
        perturbation_strength = 0.02
        for idx in selected_indices:
            y, x = patch_positions[idx]
            patch = result[y:y + patch_size, x:x + patch_size]
            blended = (1.0 - perturbation_strength) * patch + perturbation_strength * watermark_normalized
            result[y:y + patch_size, x:x + patch_size] = np.clip(blended, 0, 1)
        
        return (result * 255).astype(np.uint8)    
    def resize_image(self, image, target_size):
        """
        Resize image to target size.
        
        Args:
            image (np.ndarray): Input image
            target_size (tuple): Target size as (height, width)
        
        Returns:
            np.ndarray: Resized image
        """
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
    
    def normalize_image(self, image):
        """
        Normalize image to [0, 1] range.
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Normalized image
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image
    
    def denormalize_image(self, image):
        """
        Convert normalized image back to [0, 255] uint8.
        
        Args:
            image (np.ndarray): Normalized image
        
        Returns:
            np.ndarray: Uint8 image
        """
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    def calculate_metrics(self, original_image, processed_image):
        """
        Calculate quality metrics between original and processed images.
        
        Args:
            original_image (np.ndarray): Original image [0, 255] or [0, 1]
            processed_image (np.ndarray): Processed image [0, 255] or [0, 1]
        
        Returns:
            dict: Metrics including MSE and PSNR
        """
        # Normalize to [0, 1] if needed
        if original_image.dtype == np.uint8:
            orig_normalized = original_image.astype(np.float32) / 255.0
        else:
            orig_normalized = original_image.astype(np.float32)
            
        if processed_image.dtype == np.uint8:
            proc_normalized = processed_image.astype(np.float32) / 255.0
        else:
            proc_normalized = processed_image.astype(np.float32)
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((orig_normalized - proc_normalized) ** 2)
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = float('inf')
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'mse_formatted': f"{mse:.6f}",
            'psnr_formatted': f"{psnr:.2f} dB" if psnr != float('inf') else "Inf dB"
        }


def load_models(encoder_path="models/encoder.h5", decoder_path="models/decoder.h5"):
    """
    Load encoder and decoder models.
    
    Args:
        encoder_path (str): Path to encoder model
        decoder_path (str): Path to decoder model
    
    Returns:
        WatermarkProcessor: Initialized processor with both models
    """
    return WatermarkProcessor(encoder_path, decoder_path)
