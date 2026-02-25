import re
from typing import List, Dict, Any
import logging
from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as md
except ImportError:
    def md(html, **kwargs): return html

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def get_token_len(text): return len(tokenizer.encode(text))
except ImportError:
    def get_token_len(text): return len(text) // 4

logger = logging.getLogger(__name__)

class ConfluenceProcessor:
    """
    Process Confluence HTML data into RAG-ready markdown chunks.
    Maintains Parent-Child relationship compatibility.
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def process_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Convert HTML to Markdown
        2. Clean text
        3. Chunking
        4. Return Parent and Child objects
        """
        # 1. HTML -> Markdown
        html_content = page_data.get('content', '')
        
        # Optional: pre-process HTML with BeautifulSoup (e.g. removing scripts/styles or complex macros)
        soup = BeautifulSoup(html_content, "html.parser")
        # remove macros or extra fluff if needed
        for tag in soup(["script", "style"]):
            tag.decompose()
            
        clean_html = str(soup)
        markdown_text = md(clean_html, heading_style="ATX")
        
        # 2. Text Cleaning
        clean_text = self._clean_text(markdown_text)
        
        # Build full formatted display text for parent
        full_text = f"# {page_data['title']}\n"
        full_text += f"**Space**: {page_data['components'][0]} | **Author**: {page_data['author']} | **Updated**: {page_data['updated']}\n"
        full_text += f"**URL**: {page_data['url']}\n\n"
        full_text += clean_text

        # 3. Chunking
        chunks = self._chunk_text(clean_text)
        
        # 4. Wrap with metadata
        chunks_list = [
            {
                "id": f"{page_data['key']}_{i}",
                "text": chunk,
                "metadata": {
                    "source": "confluence",
                    "key": page_data['key'],
                    "url": page_data['url'],
                    "title": page_data['title'],
                    "updated": page_data['updated'],
                    "components": page_data.get('components', []),
                    "parent": page_data.get('parent'),
                    "chunk_index": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
        parent_doc = page_data.copy()
        # Remove raw HTML from parent to save space
        if "content" in parent_doc:
            del parent_doc["content"]
            
        parent_doc["full_formatted_text"] = full_text
        
        return {
            "parent": parent_doc,
            "chunks": chunks_list
        }

    def _clean_text(self, text: str) -> str:
        """Apply filters or regex rules to wiki output."""
        # Replace Confluence image tags
        text = re.sub(r'!([^!]+)!', r' [IMAGE: \1] ', text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """Semantic chunking with headers and paragraphs, including overlap."""
        chunks = []
        sections = re.split(r'(^#+ .+$)', text, flags=re.MULTILINE)
        
        current_chunk = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            if get_token_len(current_chunk) + get_token_len(section) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    overlap_chars = self.overlap * 4
                    overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                    first_space = overlap_text.find(' ')
                    if first_space != -1:
                        overlap_text = overlap_text[first_space:]
                    
                    current_chunk = "..." + overlap_text + "\n\n" + section
                else:
                    # Section itself is larger than max_chunk_size
                    # Fallback to paragraph splitting
                    paragraphs = section.split('\n\n')
                    for p in paragraphs:
                        if get_token_len(current_chunk) + get_token_len(p) > self.max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                                first_space = overlap_text.find(' ')
                                if first_space != -1:
                                    overlap_text = overlap_text[first_space:]
                                current_chunk = "..." + overlap_text + "\n\n" + p
                            else:
                                current_chunk = p
                        else:
                            current_chunk += "\n\n" + p
            else:
                current_chunk += "\n" + section

        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
