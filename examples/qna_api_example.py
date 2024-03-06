import uvicorn

from fastapi import (
    FastAPI,
    Body,
    HTTPException,
)

from rag_doc_search_template import config_init, get_bot_instance
from rag_doc_search_template.src.models.user_prompt import UserPrompt

from rag_doc_search_template.utils.miscellaneous import get_logger

from dotenv import load_dotenv
import os 

root_path = "path to your env file"
dotenv_path = os.path.join(root_path, ".env")
load_dotenv(dotenv_path)


app = FastAPI()
logger = get_logger()

# add your config here in config_init
config = config_init({})

bot_model = get_bot_instance(config.ai_provider)

qa_instance = bot_model.create_qa_instance()


@app.post("/api/qna")
async def process_prompt(payload: UserPrompt = Body()):
    if len(payload.prompt.strip()) <= 0:
        raise HTTPException(
            status_code=400, detail="Prompt can not be an empty string with white space"
        )
    try:
        logger.info(f"Question asked: {payload.prompt}")
        result = qa_instance.invoke(input=payload.prompt)
        logger.info(f"Answer generated: {result['result']}")
        return {"output": result["result"]}
    except Exception as e:
        logger.error(f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}")

uvicorn.run(app, host="0.0.0.0", port=8000)
