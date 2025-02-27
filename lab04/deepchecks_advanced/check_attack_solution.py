from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.vision import SingleDatasetCheck, VisionData
from deepchecks.vision.context import Context
import plotly.express as px
import torch
import torch.nn.functional as F

from typing import Any


class FGSMAttackCheck(SingleDatasetCheck):
    """A check to test if the model is robust to FGSM attacks."""

    # TODO: Add epsilon and model as a parameter.
    def __init__(
        self, device: str = "cpu", epsilon: float = 1e-4, model=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.epsilon = epsilon
        self.model = model

    # You can ignore the following method, we don't need it for this check, except to initialize the cache.
    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        # Initialize cache. You can use a different data structure if you want.
        self.cache = {}

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        model = self.model

        batch = zip(batch.original_images, batch.original_labels)
        for sample in batch:
            x, y = sample
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            x.requires_grad = True
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            y = y.reshape(1)

            x = x.permute(2, 0, 1)
            x = x.reshape(1, 3, 224, 224)

            logits = model(x)
            y_hat = F.softmax(logits, dim=1)

            loss = F.nll_loss(y_hat, y)
            model.zero_grad()
            loss.backward(inputs=[x])

            # TODO: Perform the FGSM attack and check if the model is fooled.
            x_tilde = self.fgsm_attack(x, self.epsilon)
            logits_tilde = model(x_tilde)
            y_tilde = torch.argmax(torch.softmax(logits_tilde, dim=1))

            if y_tilde != y.int():
                self.cache["failing_samples"] = self.cache.get("failing_samples", 0) + 1
            self.cache["total_samples"] = self.cache.get("total_samples", 0) + 1

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        # TODO: Get the results from the cache and compute the ratio of failing samples.
        result = {
            "ratio": self.cache["failing_samples"] / self.cache["total_samples"],
        }

        # TODO: Create a plotly express figure to display the ratio of failing samples.
        # (yes, you have my permission create a pie chart: https://plotly.com/python/pie-charts/)
        sizes = [result["ratio"], 1 - result["ratio"]]
        labels = ["Failing", "Not failing"]
        fig = px.pie(values=sizes, names=labels, title="Ratio of failing samples")

        # Pass the plotly figure to the display variable.
        display = [fig]
        return CheckResult(result, display=display)

    @staticmethod
    def fgsm_attack(x, epsilon):
        # TODO: Implement the FGSM attack.
        x_tilde = x + epsilon * x.grad.data.sign()
        x_tilde = torch.clamp(x, 0, 1)
        return x_tilde
