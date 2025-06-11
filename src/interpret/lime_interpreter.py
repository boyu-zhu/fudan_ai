import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, mark_boundaries
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import logging
import gradio as gr
import os
from datetime import datetime
from PIL import Image

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class LimeFudan:
    def __init__(self, image: np.ndarray, classes: dict[int, str], classifier_fn,
                 num_samples=1000, kernel_width=None, random_state=None, alpha=1.0):
        self.image = image
        self.classes = classes
        self.classifier_fn = classifier_fn
        self.num_samples = num_samples
        self.replacement_color = image.mean(axis=(0, 1))  # mean color
        self.random_state = np.random.RandomState(random_state)
        self.kernel_width = kernel_width
        self.alpha = alpha

    def explain(self):
        logger.info("Starting explanation process.")
        self.generate_superpixels()
        self.generate_dataset()
        self.calculate_weights()
        self.train_linear_model()
        logger.info("Explanation process completed.")

    def generate_superpixels(self):
        logger.info("Generating superpixels...")
        self.superpixels = quickshift(self.image, kernel_size=4, max_dist=200, ratio=0.2)
        self.num_superpixels = np.unique(self.superpixels).shape[0]
        logger.info(f"Found {self.num_superpixels} superpixels.")

    def generate_dataset(self):
        logger.info("Generating perturbations...")
        features = []
        predictions = []

        for i in tqdm(range(self.num_samples)):
            mask = self.random_state.binomial(1, p=self.random_state.uniform(0.2, 0.8), size=self.num_superpixels)
            if i == 0:
                mask = np.ones(self.num_superpixels)  # Original image as first sample
            features.append(mask)
            perturbed_img = self.apply_mask(mask)
            pred = self.classifier_fn(perturbed_img).squeeze()
            predictions.append(pred)

        self.features = np.array(features)
        self.predictions = np.array(predictions)

        avg_preds = self.predictions.mean(axis=0)
        self.top_label = int(np.argmax(avg_preds))
        logger.info(f"Top predicted label: {self.top_label} ({self.classes.get(self.top_label, 'Unknown')})")

    def apply_mask(self, mask):
        img = self.image.copy()
        for sp in range(self.num_superpixels):
            if mask[sp] == 0:
                img[self.superpixels == sp] = self.replacement_color
        return img

    def calculate_weights(self):
        logger.info("Calculating sample weights...")
        distances = pairwise_distances(self.features, np.ones((1, self.num_superpixels)), metric='cosine').ravel()
        if self.kernel_width is None:
            self.kernel_width = np.sqrt(self.num_superpixels) * 0.75
        self.sample_weights = np.exp(-(distances ** 2) / self.kernel_width ** 2)
        logger.info("Sample weights computed.")

    def train_linear_model(self, num_features=80, positive_only=True):
        logger.info("Training linear model...")
        model = Ridge(alpha=self.alpha)
        y = self.predictions[:, self.top_label]
        model.fit(self.features, y, sample_weight=self.sample_weights)
        self.linear_model = model

        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

        top_indices = np.argsort(np.abs(model.coef_))[-num_features:]
        if positive_only:
            top_indices = [i for i in top_indices if model.coef_[i] > 0]
        logger.info(f"Top contributing superpixels: {top_indices}")

        mask = np.zeros(self.num_superpixels)
        mask[top_indices] = 1
        self.explanation_mask = mask
        self.explanation_image = self.apply_mask(mask)

    def show_explanation(self, positive_only=True):
        logger.info("Showing explanation.")
    
        # Generate timestamp for unique subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder = os.path.join("explanation_logs", timestamp)
        os.makedirs(subfolder, exist_ok=True)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Class: {self.classes.get(self.top_label, str(self.top_label))}')
    
        ax1.imshow(self.image)
        ax1.set_title('Original')
    
        # Create heatmap image
        mask = self.explanation_mask
        mask_img = self.apply_mask(mask)
        boundary_img = mark_boundaries(mask_img / 255.0, self.superpixels)
    
        # Save the mask image
        mask_path = os.path.join(subfolder, "mask.png")
        Image.fromarray(mask_img.astype(np.uint8)).save(mask_path)
    
        # Save the boundary image
        boundary_path = os.path.join(subfolder, "boundary.png")
        Image.fromarray((boundary_img * 255).astype(np.uint8)).save(boundary_path)
    
        # Save the composite figure
        fig_path = os.path.join(subfolder, "explanation.jpg")
        plt.savefig(fig_path)
        plt.show()
    
        # Store paths for later use
        self.saved_images = {
            'original': self.image,
            'mask': mask_path,
            'boundary': boundary_path,
            'composite': fig_path,
            'timestamp': timestamp
        }
    
    def get_feature_importance(self):
        return self.coef_


    def show_gradio_interface(self):
        """Create a Gradio interface using saved explanation images."""

        from PIL import Image

# 读取图片
        image = Image.open("/data/zhuboyu/interpret/explanation_logs/20250602_200729/boundary.png")  # 替换为你的图片路径
        self.saved_images['boundary'] = image

        # Check if we have saved images
        if not hasattr(self, 'saved_images'):
            raise ValueError("Please run show_explanation() first to generate images")



        # Create prediction text
        pred_text = (f"Prediction: {self.classes.get(self.top_label, str(self.top_label))}\n")

        with gr.Blocks(title="Model Explanation") as interface:
            gr.Markdown("## Visual Flow")
            gr.Markdown("### LIME")
            with gr.Row():
                # Left column - Original image and prediction
                with gr.Column():
                    gr.Image(self.saved_images['original'], label="Original Image")
                    gr.Textbox(pred_text, label="Model Prediction")
                    # gr.Textbox("school bus", label="LIME Explanation Class")

                # Right column - Explanation visualizations
                with gr.Column():
                    gr.Image(self.saved_images['boundary'], label="LIME Explanation")


                    # Add download buttons
                    with gr.Row():
                        gr.Button("Download Explanation").click(
                            fn=lambda: self.saved_images['composite'],
                            outputs=gr.File(label="Download")
                        )

            # Add some styling
            interface.css = """
            .gradio-container { max-width: 1200px !important; }
            .download-btn { margin: 5px; }
            """

        return interface

