import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_model_with_shap(model, X_train, X_test, feature_names=None, sample_size=100):
    if not isinstance(X_train, pd.DataFrame):
        if feature_names is None:
            raise ValueError("feature_names required if X is not a DataFrame")
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test.iloc[:sample_size])
    shap.plots.beeswarm(shap_values)
    shap.plots.bar(shap_values)
    return shap_values
