import torch
from captum.attr import IntegratedGradients
import shap


class ExplainabilityAnalyzer:
    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(self.model)

    def integrated_gradients_explanation(self, data, target_class):
        self.model.eval()
        input_x = data.x.requires_grad_()
        input_edge_index = data.edge_index

        def forward_func(x):
            return self.model(x, input_edge_index)

        attributions = self.ig.attribute(
            input_x, target=target_class, n_steps=100)
        return attributions

    def shap_explanation(self, data, background_data):
        self.model.eval()
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(data.x)
        return shap_values

    def interpret_attributions(self, attributions, feature_names):
        attr_sum = attributions.sum(dim=0)
        attr_normalized = attr_sum / torch.norm(attr_sum)

        feature_importance = [(name, importance.item())
                              for name, importance in zip(feature_names, attr_normalized)]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return feature_importance[:10]  # Return top 10 most important features
