import gradio as gr
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import io
import base64

# Load the SAM model
MODEL_TYPE = "vit_h"  # Change as needed, e.g., "vit_b", "vit_l"
MODEL_CHECKPOINT = "<path_to_sam_model_checkpoint>"  # Provide the checkpoint path
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
mask_generator = SamAutomaticMaskGenerator(sam)

def segment_image(image, segmentation_mode):
    """
    Segment the image using SAM and return the segmentation result.

    Args:
        image: The input image (PIL Image).
        segmentation_mode: 'single_label' for one label or 'multi_label' for multiple labels.

    Returns:
        tuple: Segmented image and masks as numpy arrays.
    """
    image_np = np.array(image)

    # Generate masks
    masks = mask_generator.generate(image_np)

    if segmentation_mode == "single_label":
        # Combine all masks into one label
        combined_mask = np.zeros_like(image_np[:, :, 0], dtype=np.uint8)
        for mask in masks:
            combined_mask[mask["segmentation"]] = 255  # Single label
        segmented_image = np.dstack([image_np, combined_mask])
        return Image.fromarray(segmented_image), combined_mask

    elif segmentation_mode == "multi_label":
        # Each mask has its own label
        multi_label_mask = np.zeros_like(image_np[:, :, 0], dtype=np.uint16)
        for i, mask in enumerate(masks):
            multi_label_mask[mask["segmentation"]] = i + 1
        segmented_image = np.dstack([image_np, multi_label_mask.astype(np.uint8)])
        return Image.fromarray(segmented_image), multi_label_mask

def download_mask(mask, file_format):
    """
    Convert the mask to the specified format and provide a downloadable file.

    Args:
        mask: The mask as a numpy array.
        file_format: Desired file format ('png', 'jpg', 'npy').

    Returns:
        Downloadable file as bytes.
    """
    if file_format == "png" or file_format == "jpg":
        mask_image = Image.fromarray(mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format=file_format.upper())
        buffer.seek(0)
        return buffer
    elif file_format == "npy":
        buffer = io.BytesIO()
        np.save(buffer, mask)
        buffer.seek(0)
        return buffer

# Define Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""# Image Segmentation with SAM
        Upload an image and select the segmentation mode to get segmented outputs.
        You can also download the mask in various formats.""")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil")
            segmentation_mode = gr.Radio(
                ["single_label", "multi_label"], 
                value="single_label", 
                label="Segmentation Mode"
            )

        with gr.Row():
            segmented_output = gr.Image(label="Segmented Image")
            mask_output = gr.Image(label="Mask Output")

        with gr.Row():
            file_format = gr.Dropdown([
                "png", "jpg", "npy"
            ], value="png", label="Download Format")
            download_button = gr.File(label="Download Mask")

        submit_button = gr.Button("Segment")

        def process_and_download(image, segmentation_mode, file_format):
            segmented_image, mask = segment_image(image, segmentation_mode)
            file = download_mask(mask, file_format)
            file_name = f"mask.{file_format}"
            return segmented_image, mask, (file, file_name)

        submit_button.click(
            fn=process_and_download,
            inputs=[input_image, segmentation_mode, file_format],
            outputs=[segmented_output, mask_output, download_button]
        )

    return demo

# Run the Gradio app
demo = gradio_interface()
demo.launch()
