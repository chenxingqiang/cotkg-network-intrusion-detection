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
import sys
from pathlib import Path
import ollama

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.config import DEFAULT_CONFIG

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
                name="qwen-max",
                api_key_env="QIANWEN_API_KEY",
                max_tokens=1500,
                temperature=0.85
            ),
            "zhipu": ModelConfig(
                name="glm-4",
                api_key_env="ZHIPU_API_KEY",
                max_tokens=1024,
                temperature=0.95
            ),
            "ollama": ModelConfig(
                name="llama2",
                api_key_env="",  # No API key needed for local deployment
                api_base="http://localhost:11434",
                max_tokens=1500,
                temperature=0.85
            )
        }

        # Initialize clients
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for available models."""
        for provider, config in self.models.items():
            try:
                if provider == "openai":
                    api_key = os.getenv(config.api_key_env)
                    if api_key:
                        self.clients[provider] = OpenAI(api_key=api_key)
                elif provider == "anthropic":
                    api_key = os.getenv(config.api_key_env)
                    if api_key:
                        self.clients[provider] = anthropic.Client(api_key=api_key)
                elif provider == "deepseek":
                    api_key = os.getenv(config.api_key_env)
                    if api_key:
                        self.clients[provider] = {
                            "api_key": api_key,
                            "api_base": config.api_base
                        }
                elif provider == "qianwen":
                    api_key = os.getenv(config.api_key_env)
                    if api_key:
                        dashscope.api_key = api_key
                        self.clients[provider] = dashscope
                elif provider == "zhipu":
                    api_key = os.getenv(config.api_key_env)
                    if api_key:
                        zhipuai.api_key = api_key
                        self.clients[provider] = zhipuai
                elif provider == "ollama":
                    try:
                        # Initialize Ollama client
                        self.clients[provider] = ollama.Client(host=config.api_base)
                        # Test connection
                        self.clients[provider].list()
                        logger.info("Initialized Ollama client successfully")
                    except Exception as e:
                        logger.warning(f"Failed to connect to Ollama server: {str(e)}")

                if provider != "ollama":  # Skip logging for Ollama as it's already logged
                    logger.info(f"Initialized {provider} client successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {provider} client: {str(e)}")

    def _call_ollama(self, client, prompt: str, config: dict) -> str:
        """Call Ollama API."""
        try:
            # Generate response using Ollama client
            response = client.generate(
                model=config.get('model', 'llama2'),
                prompt=prompt,
                options={
                    'temperature': config.get('temperature', 0.85),
                    'top_p': config.get('top_p', 0.8),
                    'num_predict': config.get('max_tokens', 1500)
                }
            )
            return response['response']

        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise

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


def generate_cot(flow_data: str, config: dict = None) -> str:
    """
    Generate chain of thought analysis using configured LLM providers.
    Returns the combined response text.
    """
    if config is None:
        config = DEFAULT_CONFIG['cot']

    provider = LLMProvider()
    responses = []

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

    try:
        providers = config['provider'] if isinstance(config['provider'], list) else [config['provider']]
        
        for provider_name in providers:
            logger.info(f"Generating response using {provider_name}")
            
            try:
                if provider_name == 'ollama':
                    result = provider._call_ollama(provider.clients['ollama'], prompt, config)
                    if isinstance(result, str):
                        responses.append(result)
                        logger.info("Successfully generated response using ollama")
                
                elif provider_name == 'qianwen':
                    result = provider._call_qianwen(provider.clients['qianwen'], prompt)
                    if isinstance(result, str):
                        responses.append(result)
                        logger.info("Successfully generated response using qianwen")
                
            except Exception as e:
                logger.error(f"Error with {provider_name}: {str(e)}")
                continue
        
        if not responses:
            raise ValueError("No successful responses from any provider")
        
        # Combine responses - for now, just use the first successful response
        # In the future, we could implement more sophisticated response combination
        return responses[0]

    except Exception as e:
        logger.error(f"Error generating responses: {str(e)}")
        raise Exception(f"Failed to generate response: {str(e)}")


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
