import numpy as np
import plotly.express as px
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models


def _style(fig):
    fig.update_layout(template="plotly_white", height=350, paper_bgcolor="rgba(255,255,255,0)",
                      plot_bgcolor="rgba(255,255,255,.65)", font=dict(color="#352246"),
                      title=dict(font=dict(size=15, color="#35145f"), x=.02), margin=dict(l=12,r=12,t=42,b=34))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="rgba(103,50,166,.10)", zeroline=False)
    return fig


class VisionEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = models.ResNet50_Weights.DEFAULT
        base = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(base.children())[:-1]).to(self.device)
        self.model.eval(); self.transform = weights.transforms()

    def extract_embeddings(self, paths):
        vectors, meta = [], []
        with torch.no_grad():
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    x = self.transform(img).unsqueeze(0).to(self.device)
                    v = self.model(x).flatten(1).cpu().numpy()[0]
                    v /= np.linalg.norm(v) + 1e-12
                    vectors.append(v); meta.append({"path": p})
                except Exception:
                    pass
        if not vectors: raise ValueError("No valid images were processed.")
        return np.vstack(vectors), meta

    def cluster(self, embeddings, k):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        fig = px.scatter(x=xy[:,0], y=xy[:,1], color=labels.astype(str), title="Visual embedding map",
                         labels={"x":"PCA 1", "y":"PCA 2", "color":"Cluster"},
                         color_discrete_sequence=["#6d3bb3", "#8f5de7", "#d04a8f", "#4c87d9", "#3ca77d", "#e69b4a"])
        return labels, _style(fig)

    def similar(self, embeddings, index, n=3):
        scores = cosine_similarity(embeddings[index:index+1], embeddings)[0]
        order = np.argsort(scores)[::-1]
        return [(int(i), float(scores[i])) for i in order if i != index][:n]
