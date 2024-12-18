import unittest
import os
from flagevalmm.models.gpt import GPT
from flagevalmm.models.hunyuan import Hunyuan
from flagevalmm.models.http_client import HttpClient
from flagevalmm.models.claude import Claude
from flagevalmm.models.gemini import Gemini


class BaseTestModel:

    def test_generate_joke(self):
        system_prompt = "Answer all questions in Chinese"
        query = "Tell me a joke about Mars and Pokemon, it should be very funny"
        messages = self.model.build_message(query, system_prompt=system_prompt)
        answer = self.model.infer(messages)

        # Add assertions to verify the result
        self.assertIsNotNone(answer)
        self.assertTrue(isinstance(answer, str))
        self.assertGreater(len(answer), 0)

    def test_generate_joke_with_image(self):
        query = "Look at this image and create a funny joke based on what you see"
        messages = self.model.build_message(query, image_paths=["assets/test_1.jpg"])
        answer = self.model.infer(messages)

        # Add assertions to verify the result
        self.assertIsNotNone(answer)
        self.assertTrue(isinstance(answer, str))
        self.assertGreater(len(answer), 0)


class TestGPTModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = GPT(
            model_name="gpt-4o", temperature=0.5, use_cache=False, stream=True
        )


class TestQwenModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = GPT(
            model_name="qwen-vl-plus",
            api_key=os.environ.get("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


class TestSetpChat(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = GPT(
            model_name="step-1v-8k",
            api_key=os.environ.get("STEP_API_KEY"),
            base_url="https://api.stepfun.com/v1",
        )


class TestYiChat(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = GPT(
            model_name="yi-vision",
            api_key=os.environ.get("YI_API_KEY"),
            base_url="https://api.lingyiwanwu.com/v1",
        )


class TestHunyuanModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = Hunyuan(model_name="hunyuan-vision", use_cache=False)


class TestClaudeModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = Claude(
            model_name="claude-3-5-sonnet-20241022", use_cache=False, stream=True
        )


class TestHttpClientModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = HttpClient(
            model_name="gpt-4o-mini",
            api_key=os.environ.get("BAAI_OPENAI_API_KEY"),
            url="https://api.openai.com/v1/chat/completions",
        )


class TestGeminiModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = Gemini(model_name="gemini-1.5-flash")


if __name__ == "__main__":
    unittest.main(defaultTest="TestGeminiModel")
