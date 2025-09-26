from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 200
    )
)

model= ChatHuggingFace(llm=llm)  

result = model.invoke("How many states are there in the country india and list their names as well")
print(result)