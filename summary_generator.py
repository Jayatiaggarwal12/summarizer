from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

class SummaryGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_summary(self, text: str) -> str:
        """Generate summary using map-reduce method"""
        map_template = """Summarize this legal document chunk:
        {docs}
        CONCISE SUMMARY:"""
        map_prompt = PromptTemplate.from_template(map_template)
        
        reduce_template = """Combine these summaries into final version:
        {doc_summaries}
        FINAL SUMMARY:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )
        
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000
        )
        
        return MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs"
        ).run(text)