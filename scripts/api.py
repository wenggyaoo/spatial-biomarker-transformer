import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

openai.api_type = "azure"
# openai.api_key = "5c18c8a8a4e247d481acba9db5b8aa6b"
# openai.api_base = "https://api.hku.hk"
openai.api_key = "Fjjmj0ZlDnUfWmiIMpZ5REJeSLElmHluUBuHKmVUrr9XCYLs1FhKJQQJ99BDACfhMk5XJ3w3AAABACOGvHrF"
openai.api_base = "https://cds-icdevai05-openai-zhenqin-swc.openai.azure.com/"
openai.api_version = "2024-06-01"
chat = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    openai_api_type=openai.api_type,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version
)