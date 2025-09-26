from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model='gpt-4')

llm.invoke("name few dishes cooked regularly in the indian households")
