import unittest
import os
from flagevalmm.models.gpt import GPT
from flagevalmm.models.hunyuan import Hunyuan
from flagevalmm.models.http_client import HttpClient
from flagevalmm.models.claude import Claude
from flagevalmm.models.gemini import Gemini
from flagevalmm.models.api_response import ApiResponse


class BaseTestModel:
    def test_generate_joke(self):
        system_prompt = "Answer all questions in Chinese"
        query = "Tell me a joke about Mars and Pokemon, it should be very funny"
        messages = self.model.build_message(query, system_prompt=system_prompt)
        answer = self.model.infer(messages)

        # Handle both ApiResponse and string returns for backward compatibility
        if isinstance(answer, ApiResponse):
            content = answer.content
        else:
            content = answer

        # Add assertions to verify the result
        self.assertIsNotNone(content)
        self.assertTrue(isinstance(content, str))
        self.assertGreater(len(content), 0)

    def test_generate_joke_with_image(self):
        query = "Look at this image and create a funny joke based on what you see"
        messages = self.model.build_message(
            query, multi_modal_data={"image": ["assets/test_1.jpg"]}
        )
        answer = self.model.infer(messages)

        # Handle both ApiResponse and string returns for backward compatibility
        if isinstance(answer, ApiResponse):
            content = answer.content
        else:
            content = answer

        # Add assertions to verify the result
        self.assertIsNotNone(content)
        self.assertTrue(isinstance(content, str))
        self.assertGreater(len(content), 0)


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
        self.model = Hunyuan(model_name="hunyuan-turbos-vision", use_cache=False)


class TestClaudeModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = Claude(
            model_name="claude-3-5-sonnet-20241022", use_cache=False, stream=True
        )


class TestHttpClientModelStream(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = HttpClient(
            model_name="qvq-max",
            api_key=os.environ.get("QWEN_API_KEY"),
            url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            stream=True,
        )


class TestHttpClientModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = HttpClient(
            model_name="gpt-4o-mini",
            api_key=os.environ.get("FLAGEVAL_API_KEY"),
            url=os.environ.get("FLAGEVAL_URL"),
        )


class TestGeminiModel(BaseTestModel, unittest.TestCase):
    def setUp(self):
        self.model = Gemini(model_name="gemini-1.5-flash")


if __name__ == "__main__":
    unittest.main(defaultTest="TestHttpClientModelStream")
