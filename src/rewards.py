def compute_reward(action, true_label):
    if action == true_label:
        return 1.0
    elif action == 0 and true_label == 1:  # predicted benign but was phishing
        return -5.0
    else:  # predicted phishing but was benign
        return -1.0
