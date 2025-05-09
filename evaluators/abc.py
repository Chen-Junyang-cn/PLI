import abc

import numpy as np
import torch
from tqdm import tqdm

# from evaluators.metric_calculators import ValidationMetricsCalculator
from evaluators.metric_calculators import ValidationMetricsCalculator
from evaluators.utils import multiple_index_from_attribute_list
from utils.metrics import AverageMeterSet


class AbstractBaseEvaluator(abc.ABC):
    def __init__(self, models, dataloaders, top_k=(1, 10, 50)):
        self.models = models
        self.test_samples_dataloader = dataloaders['samples']
        self.test_query_dataloader = dataloaders['query']
        self.top_k = top_k if type(top_k) is tuple else tuple([int(k) for k in top_k.split(",")])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attribute_matching_matrix = None
        self.ref_matching_matrix = None

    def evaluate(self):
        all_results = {}
        all_test_features, all_test_attributes = self.extract_test_features_and_attributes()
        all_composed_query_features, all_query_attributes, all_ref_attributes = \
            self.extract_query_features_and_attributes()

        # Make sure test_loader is not shuffled! Otherwise, this will be incorrect
        if self.attribute_matching_matrix is None:
            self.attribute_matching_matrix = self._calculate_attribute_matching_matrix(all_query_attributes,
                                                                                       all_test_attributes)
            # print("attribute matching matrix shape", self.attribute_matching_matrix.shape)
        if self.ref_matching_matrix is None:
            self.ref_matching_matrix = self._calculate_attribute_matching_matrix(all_ref_attributes,
                                                                                 all_test_attributes)
        # print("calculate the similarity matrix", all_composed_query_features.shape, all_test_features.shape)
        similarity_matrix = self._calculate_similarity_matrix(all_composed_query_features, all_test_features)
        # attribute and ref matrix is tensor in cpu, out of cpu memory
        # print("similarity matrix shape", similarity_matrix.shape)
        recall_calculator = ValidationMetricsCalculator(similarity_matrix, self.attribute_matching_matrix,
                                                        self.ref_matching_matrix, self.top_k)
        recall_results = recall_calculator()
        all_results.update(recall_results)
        print(all_results)

        return all_results

    @abc.abstractmethod
    def _extract_image_features(self, images):
        raise NotImplementedError

    @abc.abstractmethod
    def _extract_composed_features(self, images, modifiers):
        raise NotImplementedError

    def extract_test_features_and_attributes(self):
        """
        Returns: (1) torch.Tensor of all test features, with shape (N_test, Embed_size)
                (2) list of test attributes, Size = N_test
        """
        self._to_eval_mode()

        dataloader = tqdm(self.test_samples_dataloader)
        all_test_attributes = []
        all_test_features = []
        with torch.no_grad():
            for batch_idx, (test_images, test_attr) in enumerate(dataloader):
                batch_size = test_images.size(0)
                test_images = test_images.to(self.device)

                features = self._extract_image_features(test_images)
                features = features.view(batch_size, -1).cpu()

                all_test_features.extend(features)
                all_test_attributes.extend(test_attr)

        return torch.stack(all_test_features), all_test_attributes

    @abc.abstractmethod
    def extract_query_features_and_attributes(self):
        """
            Returns: (1) torch.Tensor of all query features, with shape (N_query, Embed_size)
                    (2) list of target attributes, Size = N_query
            """
        raise NotImplementedError("Please implement extract_query_features_and_attributes method")

    def _to_eval_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].eval()

    def _calculate_recall_at_k(self, most_similar_idx, all_test_attributes, all_target_attributes):
        average_meter_set = AverageMeterSet()

        for k in self.top_k:
            k_most_similar_idx = most_similar_idx[:, :k]
            for i, row in enumerate(k_most_similar_idx):
                most_similar_sample_attributes = multiple_index_from_attribute_list(all_test_attributes, row)
                target_attribute = all_target_attributes[i]
                correct = 1 if target_attribute in most_similar_sample_attributes else 0
                average_meter_set.update('recall_@{}'.format(k), correct)
        recall_results = average_meter_set.averages()
        return recall_results

    @staticmethod
    def _calculate_attribute_matching_matrix(all_query_attributes, all_test_attributes):
        all_query_attributes, all_test_attributes = np.array(all_query_attributes).reshape((-1, 1)), \
            np.array(all_test_attributes).reshape((1, -1))
        return torch.tensor(all_test_attributes == all_query_attributes)


    def _calculate_similarity_matrix(self, composed_query_features: torch.tensor, test_features: torch.tensor) -> torch.tensor:
        """
        query_features = torch.tensor. Size = (N_test_query, Embed_size)
        test_features = torch.tensor. Size = (N_test_dataset, Embed_size)
        output = torch.tensor, similarity matrix. Size = (N_test_query, N_test_dataset)
        """
        # pair-wise cosine similarity, avoid feature not normalized, cosine_similarity will out of memory
        # similarity_matrix = torch.nn.functional.cosine_similarity(composed_query_features.unsqueeze(1),
        #                                                           test_features.unsqueeze(0), dim=2)
        norm_composed_query_features = torch.nn.functional.normalize(composed_query_features, dim=-1).float()
        norm_test_features = torch.nn.functional.normalize(test_features, dim=-1).float()
        similarity_matrix = torch.matmul(norm_composed_query_features, norm_test_features.transpose(0, 1))
        return similarity_matrix
