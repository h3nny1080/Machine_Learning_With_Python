import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

app = FastAPI(title="EcoSystem Intelligence API")

class QueryRequest(BaseModel):
    user_input: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # Initialize the state with the user's question
        initial_state = {"messages": [HumanMessage(content=request.user_input)]}
        
        # Run the LangGraph agent
        result = await app_agent.ainvoke(initial_state)
        
        # Return the last message in the chain (the AI response)
        final_answer = result["messages"][-1].content
        return {"response": final_answer}
    
    except Exception as e:
        print(f"Key loaded: {os.getenv('OPENAI_API_KEY')[:5]}...")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    load_dotenv()  # This finds your .env file and loads the key
    uvicorn.run(app, host="0.0.0.0", port=8000)
