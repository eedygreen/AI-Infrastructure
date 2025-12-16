
import json, torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from imc.utils import Checkpoint
from torchvision import models
from rich.console import Console
from imc.processor import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt

console = Console()
class PredictionResult:
    """
        Class to hold prediction results
    """
    def __init__(
        self, 
        image_path: str,
        top_classes: List[str],
        top_probs: List[float],
        true_label: Optional[str]=None
    ):
        self.image_path = image_path
        self.top_classes = top_classes
        self.top_probs = top_probs
        self.true_label = true_label

    def is_correct(self, rank: int = 1) -> bool:
        """
            Check if the top predicted class matches the true label

            Args:
                rank: Rank of prediction to check 

            Returns:
                True if correct, False if incorrect, None if true_label is not set
        """
        if self.true_label is None:
            raise ValueError("Cannot predict result without true_label.")

        return self.true_label in self.top_classes[:rank]
    
    def top1_match(self) -> bool:
        """
            Check if the top predicted class matches the true label

            Returns:
                True if correct, False if incorrect
        """
        return self.is_correct(rank=1)
    
    def topk_match(self, k: int) -> bool:
        """
            Check if the true label is within the top-k predicted classes

            Returns:
                True if correct, False if incorrect
        """
        return self.is_correct(rank=k)
        

    def __repr__(self) -> str:
        result = f"PredictionResult(image={Path(self.image_path).name})\n"
        result += f"Top Prediction: top_classes={self.top_classes[0]}, top_probs={self.top_probs[0]:.2%})\n"

        result += f"\nTop-{len(self.top_classes)} predictions: \n"
        for cls, prob in zip(self.top_classes, self.top_probs):
            if self.true_label:
                is_match = "✓" if cls == self.true_label else "✗"
                result += f"  {is_match}  Class: {cls}, Probability: {prob:.2%}\n"
            else:
                result += f"   Class: {cls}: Probabilty: {prob:.2%}\n"

        if self.true_label:
            match_status = "✓ CORRECT" if self.top1_match() else "✗ INCORRECT"
            result += f" True label: {self.true_label} [{match_status}]"

        return result

def run_predict(
    image_path: Union[str, List[str]], 
    checkpoint_path: str,
    gpu: Optional[bool], 
    topk=5,
    show_plot: Optional[bool]= True,
    save_plot: Optional[str] = None,
    category_names: Optional[str]=None,
    class_to_idx: Optional[Dict[str, int]] = None
    ) -> Union[PredictionResult, List[PredictionResult]]:
    """
        Predict the class (or classes) of an image using a trained deep learning model.\n
        
        Args:\n
            image_path: Path to image file or list of image file paths

            checkpoint_path: Trained PyTorch model for inference

            topk: Number of top most likely classes to return

            gpu: Optional: Disable (default)

            category_names: Optional path to JSON file mapping categories to real names

        Returns:
            PredictionResult or list of PredictionResult: Contains top classes and probabilities

        Usage:\n
            python -m imc predict --image-path <path_to_image> --model-path <path_to_model> --topk 5 \n
            python -m imc predict --image-path <path_to_image1> <path_to_image2> --model-path <path_to_model> --topk 3 \n
    """
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model = Checkpoint.load_checkpoint(checkpoint_path, gpu)
    model.to(device)
    model.eval()

    if category_names:
        with open(category_names, 'r') as f:
            class_to_name = json.load(f)
        idx_to_class = {v: class_to_name[k] for k, v in class_to_idx.items()}

    else:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    feature_extraction = models.vgg16(pretrained=True).features
    feature_extraction.to(device)
    feature_extraction.eval()

    image_paths = _get_image_paths(image_path)

    results = []

    for img_path in image_paths:
        result = _predict_single_image(
            image_path=img_path,
            model=model,
            topk=topk,
            device=device,
            feature_extraction=feature_extraction,
            idx_to_class=idx_to_class
        )
        results.append(result)
        console.print(result)

    if show_plot or save_plot:
        if len(results) == 1:
            console.print("[blue]Displaying prediction visualization...")
            plot_predict(results[0], save_path=save_plot)
            if show_plot:
                plt.show()
        else:
            console.print(f"[blue]Displaying batch predictions for {len(results)} images...")
            plot_batch_predict(results, max_display=min(9, len(results)), save_path=save_plot)
            if show_plot:
                plt.show()

    return results[0] if len(results) == 1 else results

def _get_image_paths(image_path: Union[str, List[str]]) -> List[str]:
    """
        Convert various input formats to List of Images paths
    """

    if isinstance(image_path, list):
        return image_path
    
    path = Path(image_path)

    if path.is_file():
        return [str(path)]
    
    if path.is_dir():
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(path.rglob(ext))
        return [str(p) for p in images]
    
    return []

def _preprocess_image(image_path: str) -> np.ndarray:
    """
        Preprocess a single image for model prediction.

        Args:
            image_path: Path to the image file

        Returns:
            np.ndarray: Preprocessed image tensor
    """
    img = Image.open(image_path).convert("RGB")

    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return preprocess(img)

def _predict_single_image(
    image_path: str,
    model: torch.nn.Module,
    topk: int,
    device: torch.device,
    feature_extraction: torch.nn.Module,
    idx_to_class: Dict[int, str]
) -> PredictionResult:
    """
        Predict the class of a single image using the trained model.

        Args:
            image_path: Path to the image file

            model: Trained PyTorch model for inference

            topk: Number of top most likely classes to return

            class_to_idx: Mapping from class indices to actual class labels
        Returns:
            PredictionResult: Contains top classes and probabilities
    """

    img_array = _preprocess_image(image_path)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = feature_extraction(img_tensor)
        features = features.view(features.size(0), -1)

        # predicts
        outputs = model(features)
        probs = torch.exp(outputs)
    
    # get top k
    top_probs, top_indices = probs.topk(topk, dim=1)
    top_probs = top_probs.cpu().squeeze().tolist()
    top_indices = top_indices.cpu().squeeze().tolist()

    if isinstance(top_probs, float):
        top_probs = [top_probs]
        top_indices = [top_indices]

    top_classes = [idx_to_class[idx] for idx in top_indices]

    result = PredictionResult(
        image_path= image_path,
        top_classes = top_classes,
        top_probs = top_probs
    )
    return result


def batch_predict(
    image_path: Union[str, List[str]],
    checkpoint_path: str,
    gpu: Optional[bool],
    topk=5,
    category_names: Optional[str]=None,
    class_to_idx: Optional[Dict[str, int]]=None,
    show_plot: Optional[bool] = True,
    save_plot: Optional[str] = None
) -> Dict:
    """
        Predict the classes of multiple images using a trained deep learning model.

        Args:
            image_path: Directory containing image files

            model: Trained PyTorch model for inference

            topk: Number of top most likely classes to return

            device: Device to perform inference on ('cpu' or 'cuda')

            category_names: Optional path to JSON file mapping categories to real names

        Returns:
            Dict: prediction results for each image
    """
    path = Path(image_path)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        image_paths.extend(path.rglob(ext))

    console.print(f"Found {len(image_paths)} images for prediction in {path}.")

    results = run_predict(
        image_path=[str(p) for p in image_paths],
        checkpoint_path=checkpoint_path,
        gpu=gpu,
        topk=topk,
        category_names=category_names,
        class_to_idx=class_to_idx,
        show_plot=show_plot,
        save_plot=save_plot
    )

    if not isinstance(results, list):
        results = [results]

    total = len(results)
    correct_top1 = sum(r.top1_match() for r in results if r.true_label)
    correct_topk = sum(r.topk_match(topk) for r in results if r.true_label)
    metrics = {
        "total_images": total,
        "correct_top1": correct_top1,
        "correct_topk": correct_topk,
        "top1_accuracy": correct_top1 / total * 100 if total > 0 else 0,
        "topk_accuracy": correct_topk / total * 100 if total > 0 else 0
    }

    console.print(f"\n[bold green]Batch Prediction Complete!")
    console.print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    console.print(f"Top-{topk} Accuracy: {metrics['topk_accuracy']:.2f}%")
    return metrics

def plot_predict(
    result: PredictionResult,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the image with a bar graph of top-k predictions.
    
    Args:
        result: PredictionResult object containing predictions
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Usage:
        result = run_predict('image.jpg', 'model.pth', gpu=True, topk=5)
        plot_prediction(result)
        plt.show()
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    

    img = Image.open(result.image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f"Image: {Path(result.image_path).name}", fontsize=12, fontweight='bold')
    
 
    classes = result.top_classes
    probs = [p * 100 for p in result.top_probs]  # Convert to percentages
    

    y_pos = np.arange(len(classes))
    bars = ax2.barh(y_pos, probs, color='steelblue', alpha=0.8)
    
    # Highlight the top prediction
    bars[0].set_color('darkgreen')
    bars[0].set_alpha(1.0)
    
    if result.true_label:
        for i, cls in enumerate(classes):
            if cls == result.true_label:
                bars[i].set_color('green' if i == 0 else 'orange')
                bars[i].set_alpha(1.0)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()  # Top prediction at the top
    ax2.set_xlabel('Probability (%)', fontsize=11)
    ax2.set_title('Top-K Predictions', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(probs) * 1.1)  # padding
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2f}%', 
                ha='left', va='center', fontsize=9)
    
    if result.true_label:
        status = "✓ CORRECT" if result.top1_match() else "✗ INCORRECT"
        color = "green" if result.top1_match() else "red"
        fig.suptitle(f"True Label: {result.true_label} [{status}]", 
                    fontsize=13, fontweight='bold', color=color, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved plot to {save_path}")
    
    return fig

def plot_batch_predict(
    results: List[PredictionResult],
    max_display: int = 6,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple predictions in a grid layout.
    
    Args:
        results: List of PredictionResult objects
        max_display: Maximum number of images to display
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Usage:
        results = run_predict(['img1.jpg', 'img2.jpg'], 'model.pth', gpu=True, topk=5)
        plot_batch_predictions(results)
        plt.show()
    """

    results = results[:max_display]
    n_images = len(results)
    
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    
    for idx, result in enumerate(results):
        ax_img = plt.subplot(n_rows, n_cols * 2, idx * 2 + 1)
        img = Image.open(result.image_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        
        if result.true_label:
            status = "✓" if result.top1_match() else "✗"
            color = "green" if result.top1_match() else "red"
            ax_img.set_title(f"{status} {Path(result.image_path).name}", 
                           fontsize=10, color=color, fontweight='bold')
        else:
            ax_img.set_title(Path(result.image_path).name, fontsize=10)
        
        ax_bar = plt.subplot(n_rows, n_cols * 2, idx * 2 + 2)
        
        classes = result.top_classes
        probs = [p * 100 for p in result.top_probs]
        
        y_pos = np.arange(len(classes))
        bars = ax_bar.barh(y_pos, probs, color='steelblue', alpha=0.7)

        bars[0].set_color('darkgreen')
        bars[0].set_alpha(1.0)
        
        if result.true_label:
            for i, cls in enumerate(classes):
                if cls == result.true_label:
                    bars[i].set_color('green' if i == 0 else 'orange')
        
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(classes, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Prob (%)', fontsize=9)
        ax_bar.set_xlim(0, max(probs) * 1.15)
        
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax_bar.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.1f}%', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Saved batch plot to {save_path}")
    
    return fig

