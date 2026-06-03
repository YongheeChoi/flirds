"""Backend builders: produce the (loss_fn, pkeys) pair the estimator/oracle
consume.  loss_fn(params, buffers) -> scalar val loss; pkeys = trainable param
names.  CNN now (backends/cnn); LLM next (backends/llm)."""
