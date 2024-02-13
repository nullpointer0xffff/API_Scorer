from abc import ABC
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class Metric(ABC):
    def accummulate(self, batch_outputs, batch_inputs):
        raise NotImplementedError
    
    def log(self):
        raise NotImplementedError


class DocMetric(Metric):
    """Given outputs of batches, calculate the per class metrics (precision, recall, F1, overall accuracy)
        Given the count of prediction class_i, how many are really class_i from ground truth
    """
    def __init__(self, num_classes=5) -> None:
        super().__init__()
        # score from 0 to 4, representing actual score 1-5
        self.num_classes = num_classes
        self.tp = [0] * num_classes
        self.fp = [0] * num_classes
        self.fn = [0] * num_classes 
    
    def accummulate(self, pred_labels, gt_labels):
        """Accumulate batches during evaluation

        Args:
            pred_labels: array of predicted labels
            gt_labels: array of ground truth labels
        """
        assert len(pred_labels) == len(gt_labels)
        
        for i in range(self.num_classes):
            self.tp[i] += ((pred_labels == i) & (gt_labels == i)).sum()
            self.fp[i] += ((pred_labels == i) & (gt_labels != i)).sum()
            self.fn[i] += ((pred_labels != i) & (gt_labels == i)).sum()
        
        
    def log(self):
        precision = [self.tp[i] / (self.tp[i] + self.fp[i]) if (self.tp[i] + self.fp[i]) > 0 else 0 for i in range(self.num_classes)]
        recall = [self.tp[i] / (self.tp[i] + self.fn[i]) if (self.tp[i] + self.fn[i]) > 0 else 0 for i in range(self.num_classes)]
        f1 = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0 for i in range(self.num_classes)]

        avg_precision = sum(precision) / self.num_classes
        avg_recall = sum(recall) / self.num_classes
        avg_f1 = sum(f1) / self.num_classes

        logger.info(f"Overall Precision: {avg_precision}")
        logger.info(f"Detail of precisions by class: \n{precision}\n")
        logger.info(f"Overall Recall: {avg_recall}")
        logger.info(f"Detail of recall by class: \n{recall}\n")
        logger.info(f"Overall F1: {avg_f1}")
        logger.info(f"Detail of F1 by class: \n{f1}\n")
        return avg_precision, avg_recall, avg_f1
            
        
        