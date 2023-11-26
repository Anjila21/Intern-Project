def pred_to_dict(text, pred, prob):
    res = {"company": ("", 0), "date": ("", 0), "address": ("", 0), "total": ("", 0)}
    keys = list(res.keys())
    seps = [0] + (np.nonzero(np.diff(pred))[0] + 1).tolist() + [len(pred)]
    for i in range(len(seps) - 1):
        pred_class = pred[seps[i]] - 1
        if pred_class == -1:
            continue

        new_key = keys[pred_class]
        new_prob = prob[seps[i]: seps[i + 1]].max()
        if new_prob > res[new_key][1]:
            res[new_key] = (text[seps[i]: seps[i + 1]], new_prob)

    return {k: regex.sub(r"[\t\n]", " ", v[0].strip()) for k, v in res.items()}


def test(model):
    model.eval()
    with torch.no_grad():
        oupt = model(text_tensor)
        prob = torch.nn.functional.softmax(oupt, dim=2)
        prob, pred = torch.max(prob, dim=2)
        prob = prob.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        real_text = etfo
        result = pred_to_dict(real_text, pred, prob)
        with open("output.json", "w", encoding="utf-8") as json_opened:
            json.dump(result, json_opened, indent=4)
        return result