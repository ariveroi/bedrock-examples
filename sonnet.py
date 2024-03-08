import boto3
import json

bedrock_rt = boto3.client("bedrock-runtime")


def create_claude_body(
    messages=[{"role": "user", "content": "Hello!"}],
    token_count=150,
    temp=0,
    topP=1,
    topK=250,
    stop_sequence=["Human"],
):
    """
    Simple function for creating a body for Anthropic Claude models.
    """
    body = {
        "messages": messages,
        "max_tokens": token_count,
        "temperature": temp,
        "anthropic_version": "",
        "top_k": topK,
        "top_p": topP,
        "stop_sequences": stop_sequence,
    }
    return body


def get_claude_response(
    messages="",
    token_count=250,
    temp=0,
    topP=1,
    topK=250,
    stop_sequence=["Human:"],
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
):
    """
    Simple function for calling Claude via boto3 and the invoke_model API.
    """
    body = create_claude_body(
        messages=messages,
        token_count=token_count,
        temp=temp,
        topP=topP,
        topK=topK,
        stop_sequence=stop_sequence,
    )
    response = bedrock_rt.invoke_model(modelId=model_id, body=json.dumps(body))
    response = json.loads(response["body"].read().decode("utf-8"))
    return response


prompt = [{"role": "user", "content": "tell me a story"}]
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
text_resp = get_claude_response(
    messages=prompt,
    token_count=250,
    temp=0,
    topP=1,
    topK=0,
    stop_sequence=["Human:"],
    model_id=model_id,
)
print(text_resp)
