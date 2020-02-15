import json
from dataclasses import dataclass, field
import pandas as pd
from collections import OrderedDict

import torch
from IPython.core.display import clear_output, display
from torch.utils.tensorboard import SummaryWriter
import time
from fiction_segmentation.Metrics import Metrics


@dataclass
class _Metrics:
    """Dataclass for storing confusion values and loss"""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    loss: float = 0.0

    def __add__(self, m):
        """Returns metrics class by adding parameters
        :param m - Tuple containing metrics in the format (tp, fp, fn, loss)
        """
        return _Metrics(self.tp + m[0], self.fp + m[1], self.fn + m[2], self.loss + m[3])


@dataclass
class _Epoch:
    count: int = 0
    boundary_metrics: _Metrics = _Metrics()
    tag_metrics: _Metrics = _Metrics()
    start_time: float = 0.0


@dataclass
class _Run:
    params: tuple = None
    count: int = 0
    data: list = field(default_factory=list)
    start_time: float = 0.0


class RunManager:
    """Calculates metrics and manages tensorboard"""

    def __init__(self):
        """Initialize objects"""
        self.epoch = _Epoch()
        self.run = _Run()
        self.data_size = None
        self.tb = None
        self.network = None

    def begin_run(self, network, run, data_size):
        """Load run parameters and tensorboard"""
        self.run.start_time = time.time()
        self.run.params = run
        self.run.count += 1
        self.data_size = data_size

        self.tb = SummaryWriter(comment=f'-{run}')
        self.network = network

    def end_run(self):
        """Closes tensorboard and resets epoch count"""
        self.tb.close()
        self.epoch.count = 0

    def begin_epoch(self):
        """Initializes epoch"""
        self.epoch.count = 0
        self.epoch.boundary_metrics.loss = 0
        self.epoch.start_time = time.time()

    def end_epoch(self):
        """Calculates results and appends to run data"""
        # Update duration
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        def fl_score(tp, fp, fn):
            """Calculate F1 score"""
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1

        # Calculate F1 scores
        boundary_f1 = Metrics.fl_score(
            self.epoch.boundary_metrics.tp,
            self.epoch.boundary_metrics.fp,
            self.epoch.boundary_metrics.fn
        )
        tag_f1 = Metrics.fl_score(self.epoch.tag_metrics.tp, self.epoch.tag_metrics.fp, self.epoch.tag_metrics.fn)

        # Calculate mean loss
        boundary_loss = self.epoch.boundary_metrics.loss / self.data_size
        tag_loss = self.epoch.tag_metrics.loss / self.data_size

        # Visualizing all stats
        self.tb.add_scalar('Boundary loss', boundary_loss, self.epoch.count)
        self.tb.add_scalar('Tag loss', tag_loss, self.epoch.count)
        self.tb.add_scalar('Boundary F1 score', boundary_f1, self.epoch.count)
        self.tb.add_scalar('Tag F1 score', tag_f1, self.epoch.count)

        # Adding histograms for gradients and weights
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        # Create results
        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["boundary_loss"] = boundary_loss
        results["tag_loss"] = tag_loss
        results["boundary_f1"] = boundary_f1
        results["tag_f1"] = tag_f1
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration
        for k, v in self.run.params._asdict().items():
            results[k] = v

        # Store to data list and create dataframe
        self.run.data.append(results)
        df = pd.DataFrame.from_dict(self.run.data, orient='columns')

        # Display dataframe
        clear_output(wait=True)
        display(df)

    def track_metrics(self, bd_loss, tag_loss, bd_preds, bd_labels, tag_preds, tag_labels):
        """Calulates counts of TP, FP and FN"""
        # Calculate confusion vectors
        bd_confusion = bd_preds / bd_labels
        tag_confusion = tag_preds / tag_labels

        self.epoch.boundary_metrics = self.epoch.boundary_metrics + (
            torch.sum(bd_confusion == 1).item(),
            torch.sum(bd_confusion == float('inf')).item(),
            torch.sum(bd_confusion == 0).item(),
            bd_loss.item()
        )

        self.epoch.tag_metrics = self.epoch.tag_metrics + (
            torch.sum(tag_confusion == 1).item(),
            torch.sum(tag_confusion == float('inf')).item(),
            torch.sum(tag_confusion == 0).item(),
            tag_loss.item()
        )

    def save(self, fileName):
        """Save results to file"""
        pd.DataFrame.from_dict(
            self.run.data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)
