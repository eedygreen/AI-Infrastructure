
import json, torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from imc.utils import Checkpoint
from torchvision import models
from rich.console import Console
from imc.processor import Compose, Resize, CenterCrop, ToTensor, Normalize

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
    class_to_idx: Optional[Dict[str, int]]=None
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
        class_to_idx=class_to_idx
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
