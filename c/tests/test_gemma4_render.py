"""render_chat_gemma4 must match the chat template embedded in the Gemma 4 GGUF byte for byte.

Needs jinja2 and a Gemma 4 GGUF (GEMMA4_GGUF=path); skipped otherwise.
"""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))

try:
    import jinja2
except ImportError:                      # pragma: no cover
    jinja2 = None

import openai_server as server

GGUF = os.environ.get("GEMMA4_GGUF", "")


def template_source():
    import gguf_reader
    reader = gguf_reader.GGUFReader(GGUF) if hasattr(gguf_reader, "GGUFReader") else None
    if reader is None:
        raise unittest.SkipTest("gguf_reader has no GGUFReader")
    return reader.metadata["tokenizer.chat_template"]


def reference(source, messages, thinking):
    def raise_exception(message):
        raise RuntimeError(message)
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False,
                             extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source).render(messages=messages, bos_token="<bos>",
                                          add_generation_prompt=True, enable_thinking=thinking)


CASES = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
     {"role": "user", "content": "Bye  "}],
    [{"role": "user", "content": "A"},
     {"role": "assistant", "content": "<|channel>thought\nhm\n<channel|>Answer"},
     {"role": "user", "content": "B"}],
]


@unittest.skipUnless(jinja2 and GGUF and os.path.exists(GGUF), "needs jinja2 and GEMMA4_GGUF")
class RenderMatchesTemplate(unittest.TestCase):
    def test_cases(self):
        source = template_source()
        server.ARCH = "gemma4"
        for messages in CASES:
            for thinking in (False, True):
                want = reference(source, messages, thinking)
                got = server.render_chat_gemma4(messages, enable_thinking=thinking)
                self.assertEqual(got, want, (messages, thinking))
                self.assertEqual(server.render_chat_for_arch(messages, thinking), want)


class RenderBasics(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(
            server.render_chat_gemma4([{"role": "user", "content": "Hi"}]),
            "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")

    def test_thinking_system(self):
        self.assertEqual(
            server.render_chat_gemma4([{"role": "user", "content": "Hi"}], enable_thinking=True),
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n")

    def test_tools_refused(self):
        with self.assertRaises(server.APIError):
            server.render_chat_gemma4([{"role": "user", "content": "Hi"}],
                                      tools=[{"type": "function", "function": {"name": "f"}}])


if __name__ == "__main__":
    unittest.main()
