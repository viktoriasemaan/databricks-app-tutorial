from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient
import base64
import io
from PIL import Image

def _get_endpoint_task_type(endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    w = WorkspaceClient()
    ep = w.serving_endpoints.get(endpoint_name)
    return ep.task

def is_endpoint_supported(endpoint_name: str) -> bool:
    """Check if the endpoint has a supported task type."""
    task_type = _get_endpoint_task_type(endpoint_name)
    supported_task_types = ["agent/v1/chat", "agent/v2/chat", "llm/v1/chat"]
    return task_type in supported_task_types

def _validate_endpoint_task_type(endpoint_name: str) -> None:
    """Validate that the endpoint has a supported task type."""
    if not is_endpoint_supported(endpoint_name):
        raise Exception(
            f"Detected unsupported endpoint type for this basic chatbot template. "
            f"This chatbot template only supports chat completions-compatible endpoints. "
            f"For a richer chatbot template with support for all conversational endpoints on Databricks, "
            f"see https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app"
        )

def _query_endpoint(endpoint_name: str, messages: list[dict], max_tokens) -> list[dict]:
    """Calls a model serving endpoint."""
    _validate_endpoint_task_type(endpoint_name)
    
    try:
        res = get_deploy_client('databricks').predict(
            endpoint=endpoint_name,
            inputs={'messages': messages, "max_tokens": max_tokens},
        )
        if "messages" in res:
            return res["messages"]
        elif "choices" in res:
            return [res["choices"][0]["message"]]
        raise Exception("This app can only run against:"
                        "1) Databricks foundation model or external model endpoints with the chat task type (described in https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#chat-completion-model-query)"
                        "2) Databricks agent serving endpoints that implement the conversational agent schema documented "
                        "in https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent")
    except Exception as e:
        # If multimodal fails, try with text-only messages
        if "image" in str(messages):
            print(f"Multimodal request failed: {e}. Trying text-only...")
            # Remove images and try again
            text_only_messages = []
            for msg in messages:
                if "image" in msg:
                    # Keep only the text content
                    text_only_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    text_only_messages.append(msg)
            
            # Try again with text-only
            res = get_deploy_client('databricks').predict(
                endpoint=endpoint_name,
                inputs={'messages': text_only_messages, "max_tokens": max_tokens},
            )
            if "messages" in res:
                return res["messages"]
            elif "choices" in res:
                return [res["choices"][0]["message"]]
        
        # If we still fail, raise the original error
        raise e

def _convert_image_to_base64(image):
    """Convert a PIL Image to base64 string."""
    if isinstance(image, Image.Image):
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        # Convert to base64
        return base64.b64encode(img_byte_arr).decode('utf-8')
    return None

def _prepare_multimodal_messages(messages):
    """Prepare messages for multimodal endpoints by converting images to base64."""
    prepared_messages = []
    
    for message in messages:
        prepared_message = message.copy()
        
        # If message contains an image, convert it to base64
        if "image" in message:
            image = message["image"]
            base64_image = _convert_image_to_base64(image)
            if base64_image:
                # Try different multimodal formats for compatibility
                try:
                    # Format 1: Standard OpenAI format
                    prepared_message = {
                        "role": message["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                    print(f"Created multimodal message with image (base64 length: {len(base64_image)})")
                except Exception as e:
                    print(f"Error creating multimodal message: {e}")
                    # Fallback to text-only
                    prepared_message = {
                        "role": message["role"],
                        "content": message["content"]
                    }
            else:
                print("Failed to convert image to base64")
        
        prepared_messages.append(prepared_message)
    
    return prepared_messages

def query_endpoint(endpoint_name, messages, max_tokens):
    """
    Query a chat-completions or agent serving endpoint
    If querying an agent serving endpoint that returns multiple messages, this method
    returns the last message
    """
    # Prepare messages for multimodal input if needed
    prepared_messages = _prepare_multimodal_messages(messages)
    
    return _query_endpoint(endpoint_name, prepared_messages, max_tokens)[-1]
