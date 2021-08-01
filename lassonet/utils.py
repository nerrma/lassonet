import matplotlib.pyplot as plt
import numpy as np


def plot_path(model, path, X_test, y_test):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    """
    n_selected = []
    score = []
    lambda_ = []
    for save in path:
        model.load(save.state_dict)
        n_selected.append(save.selected.sum())
        score.append(model.score(X_test, y_test))
        lambda_.append(save.lambda_)

    plt.figure(figsize=(12, 12))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, score, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("score")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, score, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()

def plot_importance(model):
    width = 0.3
    importances = model.feature_importances_.numpy()
    #importances = importances/np.linalg.norm(importances)
    plt.barh(np.arange(len(importances)),np.sort(importances), width)
    plt.title("Feature importances")

    plt.yticks(np.arange(len(importances)), np.argsort(importances))
    plt.xticks(np.linspace(0, max(importances), 5))

    plt.xlabel(r"Historic importance based on $\lambda$")
    plt.ylabel("Features")

    plt.grid(linewidth=1.1)

    ag_sort = np.argsort(importances)
    for i in np.argsort(importances):
        j = ag_sort[i] 
        plt.text(importances[j],i, int(importances[j]))

    plt.tight_layout()
    plt.show()

# Define and plot normalised feature importances for all models which allow it
def plt_imp(imp, name):
    imp = np.array(imp)

    # Normalise importances
    imp = imp/np.linalg.norm(imp)
    width = 0.3

    # Plot importances similar to XGBoost
    plt.barh(np.arange(len(imp)),np.sort(imp), width)
    plt.yticks(np.arange(len(imp)), np.argsort(imp))
    plt.title(f"{name} Feature importance")
    
    plt.xticks(np.linspace(0, max(imp), 5))
    plt.ylabel("Feature")
    plt.xlabel("Normalised Importance")
    
    plt.grid(linewidth=1.1)
    
    ag_sort = np.argsort(imp)
    for i in np.argsort(imp):
        j = ag_sort[i]
        plt.text(imp[j],i, str(round(imp[j],2)))
        
    plt.show()

