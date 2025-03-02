import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackerTorch(nn.Module):
    def __init__(self,
                 max_size=10000,
                 threshold=0.6,
                 margin=0.3,
                 track=True,
                 anchor_size=100000):
        super().__init__()
        self.stored_v = None
        self.corresponding_v = None

        self.threshold = torch.tensor(threshold)
        self.margin = margin
        self.track = track

        self.max_size = max_size

        magnitude = np.random.normal(loc=23.79, scale=2.19, size=(anchor_size, 1)).astype("float32")
        anchors = np.random.normal(size=(anchor_size, 512)).astype("float32")
        anchors = anchors / np.linalg.norm(anchors, ord=2, axis=-1, keepdims=True)
        anchors *= magnitude
        anchors = torch.as_tensor(anchors)

        anchors_normalized = anchors / torch.norm(anchors, dim=1, keepdim=True)

        self.register_buffer("anchors", anchors)
        self.register_buffer("anchors_normalized", anchors_normalized)

    def forward(self, id_vectors):
        if self.stored_v is None or not self.track:
            id_vectors_normalized = id_vectors / torch.norm(id_vectors, dim=1, keepdim=True)

            id_vectors_normalized = self._check_same_similarity(id_vectors_normalized)

            if self.track:
                self.stored_v = id_vectors_normalized

            cos_d = 1 - torch.matmul(id_vectors_normalized, self.anchors_normalized.t())

            d = torch.abs(cos_d - self.margin)

            idx = torch.argmin(d, dim=1)

            anonymized_vector = self.anchors[idx]

            if self.track:
                self.corresponding_v = anonymized_vector

            return anonymized_vector
        else:
            id_vectors_normalized = id_vectors / torch.norm(id_vectors, dim=1, keepdim=True)
            id_vectors_normalized = self._check_same_similarity(id_vectors_normalized)

            b, z = id_vectors_normalized.shape

            cos_d = 1 - torch.matmul(id_vectors_normalized, self.stored_v.t())

            stored_idx = torch.argmin(cos_d, dim=1)
            stored_dis = torch.min(cos_d, dim=1).values

            keep_mask = stored_dis < self.threshold
            reject_mask = stored_dis >= self.threshold
            keep_stored_idx = torch.masked_select(stored_idx, keep_mask)

            keep_stored = self.corresponding_v[keep_stored_idx]

            # If all vectors matches, no need to do any additional compute
            if torch.sum(keep_mask) == b:
                return keep_stored
            else:
                cos_d = 1 - torch.matmul(id_vectors_normalized, self.anchors_normalized.t())

                d = torch.abs(cos_d - self.margin)

                idx = torch.argmin(d, dim=1)

                anonymized_vector = self.anchors[idx]
                anonymized_vector[keep_mask] = keep_stored

                self.corresponding_vectors = torch.concat([self.corresponding_v, anonymized_vector[reject_mask]], dim=0)
                self.stored_vectors = torch.concat([self.stored_v, id_vectors_normalized[reject_mask]], dim=0)

                return anonymized_vector

    def _check_same_similarity(self, id_vectors_normalized):
        # if there is duplicate identities in a batch, we solve this by averaging the identity vectors.
        b, z = id_vectors_normalized.shape
        cos_d = 1 - torch.matmul(id_vectors_normalized, id_vectors_normalized.t())

        similarity_matrix_mask = cos_d < self.threshold
        match_per_sample = similarity_matrix_mask.sum(0)

        id_vectors_normalized_proj = id_vectors_normalized.unsqueeze(0).repeat(b, 1, 1)
        id_vectors_normalized_proj_masked = id_vectors_normalized_proj * similarity_matrix_mask.unsqueeze(-1)

        adjusted_vectors = id_vectors_normalized_proj_masked.sum(1) / match_per_sample.unsqueeze(-1)

        return adjusted_vectors

    def slerp(self, p0, p1, t):
        p0_n = p0 / np.linalg.norm(p0, axis=-1, keepdims=True)
        p1_n = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)

        omega = np.arccos(np.sum(p0_n * p1_n, axis=-1))
        so = np.sin(omega)

        s0 = np.sin((1.0 - t) * omega) / so
        s1 = np.sin(t * omega) / so

        s0 = np.expand_dims(s0, -1)
        s1 = np.expand_dims(s1, -1)

        return s0 * p0 + s1 * p1

    def clear(self):
        self.stored_v = None
        self.corresponding_v = None
