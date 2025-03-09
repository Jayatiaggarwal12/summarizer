import difflib

class DocumentComparator:
    def compare_documents(self, doc1: str, doc2: str) -> str:
        """Compare two documents with highlighted differences"""
        differ = difflib.HtmlDiff(wrapcolumn=80)
        html_diff = differ.make_file(
            doc1.splitlines(), 
            doc2.splitlines(),
            context=True,
            numlines=2
        )
        return html_diff