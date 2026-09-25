"""Minimal client for an OpenAI-compatible chat completions server (vLLM), standard library only."""
import json
import math
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base_url, model, template_kwargs=None, timeout=600, retries=4):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        # Passed to the chat template, e.g. {"enable_thinking": False} for Qwen.
        self.template_kwargs = template_kwargs or {}
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages, max_tokens, top_logprobs=0, template_kwargs=None, structured=None):
        """Greedy decoding. Returns (content, logprobs), logprobs being the list of per-token
        {"token", "logprob", "top_logprobs"} dicts (empty unless top_logprobs > 0).
        structured: vLLM constrained decoding, e.g. {"choice": [...]}, {"regex": ...} or {"json": schema}."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "chat_template_kwargs": {**self.template_kwargs, **(template_kwargs or {})},
        }
        if top_logprobs:
            body |= {"logprobs": True, "top_logprobs": top_logprobs}
        if structured:
            body["structured_outputs"] = structured
        data = json.dumps(body).encode()
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, data, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    choice = json.load(resp)["choices"][0]
                content = choice["message"]["content"] or ""
                logprobs = (choice.get("logprobs") or {}).get("content") or []
                return content, logprobs
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
                if attempt == self.retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))


def _word(token):
    return token.strip().strip('"').strip().lower()


def token_at(logprobs, index):
    """Position of the token that covers character `index` of the generated text."""
    end = 0
    for i, pos in enumerate(logprobs):
        end += len(pos["token"])
        if end > index:
            return i
    return None


def p_yes(logprobs, yes=("yes",), no=("no",)):
    """P(yes) / (P(yes) + P(no)) at the first generated token whose text is a yes/no answer,
    using the top alternatives of that token. None if no such token was generated."""
    for pos in logprobs:
        if _word(pos["token"]) not in yes + no:
            continue
        py = pn = 0.0
        for alt in pos.get("top_logprobs") or [pos]:
            t = _word(alt["token"])
            if t in yes:
                py += math.exp(alt["logprob"])
            elif t in no:
                pn += math.exp(alt["logprob"])
        return py / (py + pn) if py + pn > 0 else None
    return None
