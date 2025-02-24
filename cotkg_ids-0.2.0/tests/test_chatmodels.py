import unittest
from unittest.mock import patch, MagicMock
import os
from http import HTTPStatus
import logging
import json

# Import your LLM code
from src.knowledge_graph.cot_generator import LLMProvider, generate_cot, parse_cot_response


class TestLLMIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_flow = """
        Source IP: 192.168.1.100
        Destination IP: 10.0.0.5
        Protocol: TCP
        Source Port: 45123
        Destination Port: 80
        Packet Count: 1000
        Bytes Transferred: 150000
        Duration: 5.2s
        """

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)

    def test_provider_initialization(self):
        """Test that each provider initializes correctly with proper API keys"""
        provider = LLMProvider()

        # Check each provider's initialization
        for model_name, config in provider.models.items():
            api_key = os.getenv(config.api_key_env)
            if api_key:
                self.assertIn(model_name, provider.clients)
                self.logger.info(f"{model_name} initialized successfully")
            else:
                self.assertNotIn(model_name, provider.clients)
                self.logger.warning(
                    f"{model_name} not initialized due to missing API key")

    def test_openai_integration(self):
        """Test OpenAI integration"""
        provider = LLMProvider()
        if "openai" not in provider.clients:
            self.skipTest("OpenAI API key not configured")

        try:
            client = provider.clients["openai"]
            response = client.chat.completions.create(
                model=provider.models["openai"].name,
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10
            )
            self.assertIsNotNone(response.choices[0].message.content)
            self.logger.info("OpenAI integration test passed")
        except Exception as e:
            # Check if error is due to API subscription
            if "quota" in str(e).lower() or "unauthorized" in str(e).lower() or "invalid" in str(e).lower():
                self.logger.warning(
                    "OpenAI test failed due to API subscription issue")
            else:
                self.fail(
                    f"OpenAI test failed due to unexpected error: {str(e)}")

    def test_anthropic_integration(self):
        """Test Anthropic integration"""
        provider = LLMProvider()
        if "anthropic" not in provider.clients:
            self.skipTest("Anthropic API key not configured")

        try:
            client = provider.clients["anthropic"]
            response = client.messages.create(
                model=provider.models["anthropic"].name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Test message"}]
            )
            self.assertIsNotNone(response.content[0].text)
            self.logger.info("Anthropic integration test passed")
        except Exception as e:
            if "unauthorized" in str(e).lower() or "invalid" in str(e).lower():
                self.logger.warning(
                    "Anthropic test failed due to API subscription issue")
            else:
                self.fail(
                    f"Anthropic test failed due to unexpected error: {str(e)}")

    def test_qianwen_integration(self):
        """Test Qianwen integration"""
        provider = LLMProvider()
        if "qianwen" not in provider.clients:
            self.skipTest("Qianwen API key not configured")

        try:
            response = provider._call_qianwen(
                provider.clients["qianwen"], "Test message")
            self.assertIsInstance(response, str)
            self.logger.info("Qianwen integration test passed")
        except Exception as e:
            if "unauthorized" in str(e).lower() or "invalid" in str(e).lower():
                self.logger.warning(
                    "Qianwen test failed due to API subscription issue")
            else:
                self.fail(
                    f"Qianwen test failed due to unexpected error: {str(e)}")

    def test_zhipu_integration(self):
        """Test Zhipu integration"""
        provider = LLMProvider()
        if "zhipu" not in provider.clients:
            self.skipTest("Zhipu API key not configured")

        try:
            client = provider.clients["zhipu"]
            response = provider._call_zhipu(client, "Test message")
            self.assertIsInstance(response, str)
            self.logger.info("Zhipu integration test passed")
        except Exception as e:
            if "unauthorized" in str(e).lower() or "invalid" in str(e).lower():
                self.logger.warning(
                    "Zhipu test failed due to API subscription issue")
            else:
                self.fail(
                    f"Zhipu test failed due to unexpected error: {str(e)}")

    def test_generate_cot_fallback(self):
        """Test the fallback mechanism of generate_cot"""
        try:
            response, provider_used = generate_cot(self.sample_flow)
            self.assertIsInstance(response, str)
            self.assertIsInstance(provider_used, str)
            self.logger.info(
                f"CoT generation successful using {provider_used}")
        except Exception as e:
            if all(["unauthorized" in str(e).lower() or "invalid" in str(e).lower() or "quota" in str(e).lower()
                    for model in ["openai", "anthropic", "qianwen", "zhipu"]]):
                self.logger.warning(
                    "All providers failed due to API subscription issues")
            else:
                self.fail(
                    f"CoT generation failed due to unexpected error: {str(e)}")

    def test_parse_cot_response(self):
        """Test the parsing of CoT responses"""
        sample_response = """
        Key Feature: High packet count
        Key Feature: Unusual port
        Likely Attack: DDoS Attack
        """

        entities, relationships = parse_cot_response(sample_response)
        self.assertTrue(len(entities) > 0)
        self.assertTrue(len(relationships) > 0)
        self.logger.info("CoT response parsing test passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
