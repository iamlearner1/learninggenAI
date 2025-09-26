from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

result = model.invoke("give me the number of states in the country india with the names and also union teritory list as well inlcude the formation dates of them as well")

print(result.content)


