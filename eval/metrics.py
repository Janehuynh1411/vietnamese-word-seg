# HMM eval metrics

def hmm_eval_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# LLM eval metrics
def llama_eval_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Hybrid eval metrics
def hybrid_eval_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
