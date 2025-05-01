"""
CLIP-based landmark matching for TOT-NAV.
Matches free-form landmark descriptions to node IDs using OpenAI CLIP.
"""
import torch
import clip
from PIL import Image
from typing import List, Dict, Any, Union


class CLIPMatcher:
    def __init__(
            self,
            graph_loader,
            top_k: int = 5
    ):
        """
        Initialize the CLIP matcher for landmark identification.

        Args:
            graph_loader: Graph loader with node images
            top_k: Number of candidate nodes to return per landmark
        """
        self.gl = graph_loader
        self.top_k = top_k

        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        # Compute image embeddings for all nodes
        self.node_image_feats = self._compute_node_image_embeddings()

    def _compute_node_image_embeddings(self) -> Dict[int, List[torch.Tensor]]:
        """
        Compute and store normalized CLIP embeddings for each image of each node.

        Returns:
            Dictionary mapping node IDs to lists of image embeddings
        """
        feats = {}
        for nid in self.gl.graph.nodes:
            imgs = self.gl.get_node_images(nid)
            if not imgs:
                continue

            embeddings = []
            for img in imgs:
                image_tensor = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_emb = self.model.encode_image(image_tensor).squeeze(0)
                    img_emb = img_emb / img_emb.norm()
                    embeddings.append(img_emb)

            feats[nid] = embeddings
        return feats

    def match(
            self,
            landmarks: Union[List[str], List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Match landmarks to nodes using CLIP embeddings.

        Args:
            landmarks: List of landmark descriptions or dicts with "landmark" keys

        Returns:
            Dictionary mapping landmarks to lists of matching nodes with confidence scores
        """
        # Extract landmark names if provided as dicts
        if landmarks and isinstance(landmarks[0], dict):
            names = [d["landmark"] for d in landmarks]
        else:
            names = landmarks  # assume List[str]

        # Create text embeddings
        prompts = [f"A picture of a {lm}" for lm in names]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

        # Match each landmark to nodes
        matches = {}
        for idx, lm in enumerate(names):
            emb = text_emb[idx]
            node_scores = []

            # Score all nodes
            for nid, img_embeds in self.node_image_feats.items():
                if not img_embeds:
                    continue
                # Get best matching image for this node
                best_score = max(float((img_emb @ emb).cpu()) for img_emb in img_embeds)
                node_scores.append((nid, best_score))

            # Sort by score
            node_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply diversity constraint - no neighboring nodes
            selected = []
            for nid, conf in node_scores:
                # Skip if this node is a neighbor of any already chosen
                if any(
                        nid == chosen
                        or self.gl.graph.has_edge(nid, chosen)
                        or self.gl.graph.has_edge(chosen, nid)
                        for chosen in selected
                ):
                    continue

                selected.append(nid)
                if len(selected) >= self.top_k:
                    break

            # Get final candidates with scores
            candidates = [(nid, score) for nid, score in node_scores if nid in selected]
            matches[lm] = [{"node_id": nid, "confidence": conf} for nid, conf in candidates]

        return matches