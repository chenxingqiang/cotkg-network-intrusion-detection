from openai import OpenAI

client = OpenAI(api_key='your-api-key')


def generate_cot(flow_data):
    prompt = f"""
    Given the following network flow data:
    {flow_data}

    Please analyze this data step by step:
    1. Identify the key features that stand out in this flow.
    2. Compare these features to known patterns of different types of network attacks.
    3. Consider any anomalies or unusual combinations of features.
    4. Hypothesize about the most likely type of network activity or attack this represents.
    5. Explain your reasoning for this hypothesis.
    6. Suggest any additional data or context that would be helpful to confirm your hypothesis.

    Based on your analysis, what type of network activity or attack do you think this flow represents?
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content


def parse_cot_response(response):
    # Parse the response to extract entities and relationships
    # This is a placeholder and would need to be implemented based on the structure of the CoT output
    entities = []
    relationships = []

    # Example parsing logic
    lines = response.split('\n')
    for line in lines:
        if 'key feature:' in line.lower():
            entities.append(
                {'type': 'Feature', 'name': line.split(':')[1].strip()})
        if 'likely attack:' in line.lower():
            entities.append(
                {'type': 'Attack', 'name': line.split(':')[1].strip()})

    # Add relationships
    for entity in entities:
        if entity['type'] == 'Feature':
            relationships.append({
                'source': entity['name'],
                'type': 'INDICATES',
                # Assume the last entity is the attack type
                'target': entities[-1]['name']
            })

    return entities, relationships
