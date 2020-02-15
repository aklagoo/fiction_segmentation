class Metrics:
    @staticmethod
    def fl_score(tp, fp, fn):
        """Calculate F1 score"""
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
