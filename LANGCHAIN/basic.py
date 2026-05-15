from langchain_groq import ChatGroq
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv
## Why StreamingStdOutCallbackHandler ??
## Because langchain will recieve tokens in streaming form but dont what to do with it.
## So it just store them in buffer and give to us in last which looks like streaming not happening
## To see streaming we are going to use StreamingStdOutCallbackhandler

load_dotenv()
llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
        )

# temperature: Controls randomness/creativity of the model's output.
#              Lower values (e.g., 0–0.3) make responses more deterministic and factual.
#              Higher values (e.g., 1–2) make responses more creative, diverse, and sometimes less accurate.
#              temperature 0 means always same output for same input

# max_tokens: Limits the maximum number of tokens (words/subwords) the model can generate in the response.
#             Helps control response length and cost. Higher value = longer output, lower value = shorter output.

result = llm.invoke("tell me a story")
print(result) ## this also prints some metadata
print(result.content) ## this only returns the result always