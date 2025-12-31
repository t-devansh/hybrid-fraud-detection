import shap
import matplotlib.pyplot as plt

def render_shap_explanation(model, raw_input):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(raw_input)

    fig, ax = plt.subplots(figsize=(8, 4))

    shap.bar_plot(
        shap_values[0],
        feature_names=raw_input.columns,
        max_display=5,
        show=False
    )

    plt.tight_layout()
    return fig
