import os
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import anthropic
from openai import OpenAI
from http import HTTPStatus
import dashscope
from dashscope import Generation
import requests
import json
import logging
import time
import zhipuai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    api_key_env: str
    api_base: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7


class LLMProvider:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {
            "openai": ModelConfig(
                name="gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY"
            ),
            "anthropic": ModelConfig(
                name="claude-3-sonnet-20240229",
                api_key_env="ANTHROPIC_API_KEY"
            ),
            "deepseek": ModelConfig(
                name="deepseek-chat",
                api_key_env="DEEPSEEK_API_KEY",
                api_base="https://api.deepseek.com/v1"
            ),
            "qianwen": ModelConfig(
                name="qwen-max",  # 可选 qwen-turbo、qwen-plus、qwen-max
                api_key_env="QIANWEN_API_KEY",
                max_tokens=1500,
                temperature=0.85
            ),
            "zhipu": ModelConfig(
                name="glm-4",
                api_key_env="ZHIPU_API_KEY",
                max_tokens=1024,
                temperature=0.95
            )
        }

        # Initialize clients
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for available models."""
        for provider, config in self.models.items():
            api_key = os.getenv(config.api_key_env)
            if api_key:
                try:
                    if provider == "openai":
                        self.clients[provider] = OpenAI(api_key=api_key)
                    elif provider == "anthropic":
                        self.clients[provider] = anthropic.Client(
                            api_key=api_key)
                    elif provider == "deepseek":
                        self.clients[provider] = {
                            "api_key": api_key,
                            "api_base": config.api_base
                        }
                    elif provider == "qianwen":
                        dashscope.api_key = api_key
                        self.clients[provider] = dashscope
                    elif provider == "zhipu":
                        # 修复智谱API初始化
                        zhipuai.api_key = api_key
                        self.clients[provider] = zhipuai
                    logger.info(f"Initialized {provider} client successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize {provider} client: {str(e)}")
            else:
                logger.warning(f"No API key found for {provider} ({config.api_key_env})")

    def _call_deepseek(self, client: Dict, prompt: str) -> str:
        """Call Deepseek API."""
        try:
            headers = {
                "Authorization": f"Bearer {client['api_key']}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.models["deepseek"].name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.models["deepseek"].max_tokens,
                "temperature": self.models["deepseek"].temperature
            }

            response = requests.post(
                f"{client['api_base']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Deepseek API error: {response.text}")

        except Exception as e:
            logger.error(f"Error calling Deepseek API: {str(e)}")
            raise

    def _call_qianwen(self, client, prompt: str) -> str:
        """Call Qianwen API."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            response = Generation.call(
                model=self.models["qianwen"].name,
                messages=messages,
                result_format='message',
                top_p=0.8,
                temperature=self.models["qianwen"].temperature,
                max_tokens=self.models["qianwen"].max_tokens
            )

            if response.status_code == HTTPStatus.OK:
                if response.output and response.output.choices:
                    result = response.output.choices[0]['message']['content']
                    return result  # 确保返回字符串
                else:
                    raise Exception("Empty response from Qianwen API")
            else:
                raise Exception(f"Qianwen API error: {response.code} - {response.message}")

        except Exception as e:
            logger.error(f"Error calling Qianwen API: {str(e)}")
            raise

    def _call_zhipu(self, client, prompt: str) -> str:
        """Call Zhipu API."""
        try:
            # 修复智谱API调用
            response = client.chat.completions.create(
                model=self.models["zhipu"].name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.models["zhipu"].temperature,
                max_tokens=self.models["zhipu"].max_tokens,
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise Exception("Empty response from Zhipu API")

        except Exception as e:
            logger.error(f"Error calling Zhipu API: {str(e)}")
            raise


def parse_cot_response(response: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse the CoT response to extract entities and relationships."""
    entities = []
    relationships = []

    try:
        # 改进解析逻辑以处理更多格式
        lines = response.split('\n')
        attack_type = None

        for line in lines:
            # 提取特征
            if any(x in line.lower() for x in ['key feature:', 'identified feature:', '- **']):
                feature = None
                if ':' in line:
                    feature = line.split(':')[1].strip()
                elif '**' in line:
                    feature = line.split('**')[1].strip()

                if feature:
                    entities.append({
                        'type': 'Feature',
                        'name': feature
                    })

            # 提取攻击类型
            if any(x in line.lower() for x in ['likely attack:', 'attack type:', 'this flow represents', 'most likely']):
                for attack_keyword in ['port scan', 'ddos', 'brute force', 'botnet', 'malware']:
                    if attack_keyword in line.lower():
                        attack_type = attack_keyword.title()
                        entities.append({
                            'type': 'Attack',
                            'name': attack_type
                        })
                        break

        # 建立关系
        if attack_type:
            for entity in entities:
                if entity['type'] == 'Feature':
                    relationships.append({
                        'source': entity['name'],
                        'type': 'INDICATES',
                        'target': attack_type
                    })

        logger.info(f"Successfully parsed response: found {len(entities)} entities and {len(relationships)} relationships")

    except Exception as e:
        logger.error(f"Error parsing CoT response: {str(e)}")

    return entities, relationships


def generate_cot(flow_data: str) -> str:
    """
    Generate chain of thought analysis using multiple LLM providers with fallback.
    Returns the response text.
    """
    provider = LLMProvider()

    prompt = f"""
    Given the following network flow data:
    {flow_data}

    Please analyze this data step by step:
    1. Identify the key features that stand out in this flow:
    2. Compare these features to known patterns of different types of network attacks:
    3. Consider any anomalies or unusual combinations of features:
    4. Hypothesize about the most likely type of network activity or attack this represents:
    5. Explain your reasoning for this hypothesis:
    6. Suggest any additional data or context that would be helpful to confirm your hypothesis:

    Based on your analysis, what type of network activity or attack do you think this flow represents?
    """

    for model_name, client in provider.clients.items():
        try:
            logger.info(f"Attempting to generate response using {model_name}")

            if model_name == "openai":
                response = client.chat.completions.create(
                    model=provider.models[model_name].name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=provider.models[model_name].max_tokens
                )
                result = response.choices[0].message.content

            elif model_name == "anthropic":
                response = client.messages.create(
                    model=provider.models[model_name].name,
                    max_tokens=provider.models[model_name].max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text

            elif model_name == "deepseek":
                result = provider._call_deepseek(client, prompt)

            elif model_name == "qianwen":
                result = provider._call_qianwen(client, prompt)

            elif model_name == "zhipu":
                result = provider._call_zhipu(client, prompt)

            # 检查并确保返回的是字符串
            if not isinstance(result, str):
                raise ValueError(f"Invalid response type from {model_name}: {type(result)}")

            logger.info(f"Successfully generated response using {model_name}")
            return result  # 只返回响应文本

        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}")
            time.sleep(1)
            continue

    raise Exception("All LLM providers failed to generate a response")


# 测试代码
if __name__ == "__main__":
    sample_flow = """
    Source IP: 192.168.1.100
    Destination IP: 10.0.0.5
    Protocol: TCP
    Source Port: 45123
    Destination Port: 80
    Packet Count: 1000
    Bytes Transferred: 150000
    Duration: 5.2s
    """

    try:
        response_text = generate_cot(sample_flow)
        print("\nResponse:")
        print(response_text)

        entities, relationships = parse_cot_response(response_text)
        print("\nExtracted Entities:", entities)
        print("\nExtracted Relationships:", relationships)

    except Exception as e:
        print(f"Failed to generate or parse response: {str(e)}")
