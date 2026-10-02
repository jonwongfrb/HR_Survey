#!/usr/bin/env python
# coding: utf-8

# In[37]:


import boto3
import logging
import streamlit as st


# In[38]:


try:
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    print("Account:", identity["Account"])
    print("Principal ARN:", identity["Arn"])
except Exception as e:
    print("Failed to get identity. Error: ", e)


# In[39]:


# Silence Streamlit's background warnings
for logger_name in logging.root.manager.loggerDict:
    if "streamlit" in logger_name:
        logging.getLogger(logger_name).disabled = True


# In[40]:


bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-gov-west-1"
)

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name="us-gov-west-1"
)

option = st.selectbox(
    "Which model do you want to use?",
    (
        "amazon.nova-lite-v1:0",
        "meta.llama3-70b-instruct-v1:0"
    )
)

max_output_tokens = st.number_input("Enter max output tokens", value=1000)
temperature = st.number_input("Enter temperature", value=0.5)

kb_id = "5PWQUMCY9M"
model_id = option
model_arn = "arn:aws:bedrock:us-gov-west--1::foundation-model/" + option
st.write(f"You selected: {option}")


# In[36]:


query = st.text_input("Ask anything about the Survey Response Dataset ")

if query:
    st.write(f"You asked \"{query}\". Working on it...")

    # 1. Retrieve relevant fragments from the managed knowledge base
    kb_response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            # Managed KBs use managedSearchConfiguration instead of vectorSearchConfiguration
            "managedSearchConfiguration": {}
        }
    )

    # Extract and combine the text snippets
    results = kb_response.get("retrievalResults", [])

    context = "\n\n".join([res.get("content", {}).get("text", "") for res in results])

    # 2. Build your custom prompt injecting the context
    prompt = f"""
    Answer the user query using the provided context.

    Context:
    {context}

    User Query: {query}
    """

    # 3. Invoke the model directly using the converse API (recommended over invoke_model)
    # This structure accepts standard temperature and maxTokens parameters cleanly
    model_response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [{"text": prompt}]
        }],
        inferenceConfig={
            "temperature": temperature,
            "maxTokens": max_output_tokens
        }
    )

    output_text = model_response["output"]["message"]["content"][0]["text"]
    # print(output_text)
    if output_text:
        st.write(f"{output_text}")
        st.write("")
        st.write("")

