from src.dependency_parse import DependencyParse


def get_metrics(predicted: DependencyParse, labeled: DependencyParse): #-> dict:
    # TODO: Your code here!
    predicted_heads = predicted.heads
    gold_heads = labeled.heads
    predicted_deprel = predicted.deprel
    gold_deprel = labeled.deprel
    head_percent = 0.0
    label_percent = 0.0
    max_heads = max(len(predicted_heads),len(gold_heads))
    for i in range(max_heads):
        if (i<len(predicted_heads) and i<len(gold_heads) and predicted_heads[i]==gold_heads[i]):
            head_percent+=1.0
    uas = head_percent/max_heads
    max_dep = max(len(predicted_deprel),len(gold_deprel))
    for i in range(max_dep):
        if (i<len(predicted_deprel) and i<len(gold_deprel) and predicted_heads[i]==gold_heads[i] and predicted_deprel[i]==gold_deprel[i]):
            label_percent+=1.0
    las = label_percent/max_dep
    return {
        "uas": uas,
        "las": las,
    }
