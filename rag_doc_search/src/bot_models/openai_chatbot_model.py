from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

from rag_doc_search.src.bot_models.chatbot_model import ChatBotModel
from rag_doc_search import config


class OpenAIChatBot(ChatBotModel):
    """
    A class representing the Bedrock ChatBot.

    This class serves as an implementation of a chatbot using the OpenAI.
    """

    def __init__(self):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=self.config.embeddings_model)
        self.vector_store = self.config.get_vector_store(embeddings=self.embeddings)
        super().__init__(
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            retriever_args=self.config.get_retriever_args(),
        )

    def create_qa_instance(self) -> RetrievalQA:
        """
        Creates and returns an instance of RetrivalQA using OpenAI Language Model.

        Returns:
        An instance of RetrivalQA.
        """
        cl_llm: BaseLanguageModel = OpenAI(
            model_name=self.config.llm,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_output_tokens,
        )
        qa = self.create_qa_chain(cl_llm)
        return qa

    def create_conversational_qa_instance(
        self, stream_handler, tracing: bool = False
    ) -> ConversationalRetrievalChain:
        """
        Creates and returns an instance of ConversationalRetrievalChain for conversational question-answering using OpenAI Language Model.

        Parameters:
        - `stream_handler`: An instance of StreamingLLMCallbackHandler used for handling streaming callbacks.
        - `tracing`: A boolean indicating whether tracing is enabled. Default is False.

        Returns:
        An instance of ConversationalRetrievalChain for conversational question-answering.
        """
        stream_manager = self.create_stream_manager(stream_handler, tracing)
        cl_llm: BaseLanguageModel = OpenAI(
            model_name=self.config.llm,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_output_tokens,
            streaming=True,
            callback_manager=stream_manager,
        )
        qa = self.create_conversational_qa_chain(cl_llm)
        return qa
