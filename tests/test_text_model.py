from text_emotion.model import predict

def test_predict_neutral_when_empty():
    res = predict("")
    assert "neutral" in res.probs and res.probs["neutral"] > 0.3
