from llama_index.logger import LlamaLogger

llama_logger = LlamaLogger()
service_context = ServiceContext.from_defaults(..., llama_logger=llama_logger)
....
response = index.query("my query")
print(llama_logger.get_logs())  # prints all logs, which basically includes all LLM inputs and responses
llama_logger.reset() 