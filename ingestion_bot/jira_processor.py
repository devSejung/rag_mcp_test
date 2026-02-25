import re
from typing import List, Dict, Any, Optional
import logging

# Optional: try to import markdownify, else use a simple fallback
try:
    from markdownify import markdownify as md
except ImportError:
    def md(html, **kwargs): return html

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def get_token_len(text): return len(tokenizer.encode(text))
except ImportError:
    def get_token_len(text): return len(text) // 4  # Rough approximation

logger = logging.getLogger(__name__)

class JiraProcessor:
    """
    Process Jira data into RAG-ready chunks.
    Handles: Markdown conversion, Log summarization, Image placeholders, Chunking.
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200, allowed_components: List[str] = None, min_length: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.allowed_components = allowed_components or []
        self.min_length = min_length

    def process_issue(self, issue_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main pipeline:
        1. Combine Description + Comments + Metadata
        2. Clean & Format (Markdown, Loqs, Images)
        3. Chunking
        """
        # 0. Check Component Filter
        if self.allowed_components:
            issue_components = issue_data.get('components', [])
            if not any(c in self.allowed_components for c in issue_components):
                logger.info(f"Skipping {issue_data['key']}: not in allowed components.")
                return None
                
        # 0. Check Content Length (Description + Comments)
        content_len = len(issue_data.get('content', '') or '')
        for comment in issue_data.get('comments', []):
            content_len += len(comment.get('body', '') or '')
            
        if content_len <= self.min_length:
            logger.info(f"Skipping {issue_data['key']}: content length ({content_len}) <= {self.min_length}.")
            return None
            
        # 1. Combine Content
        full_text = self._build_full_text(issue_data)
        
        # 2. Text Cleaning
        clean_text = self._clean_text(full_text)
        
        # 3. Chunking
        chunks = self._chunk_text(clean_text)
        
        # 4. Attach Metadata to chunks
        chunks_list = [
            {
                "id": f"{issue_data['key']}_{i}",
                "text": chunk,
                "metadata": {
                    "source": "jira",
                    "key": issue_data['key'],
                    "url": issue_data['url'],
                    "title": issue_data['title'],
                    "status": issue_data['status'],
                    "updated": issue_data['updated'],
                    "components": issue_data.get('components', []),
                    "parent": issue_data.get('parent'),
                    "chunk_index": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
        parent_doc = issue_data.copy()
        parent_doc["full_formatted_text"] = full_text
        
        return {
            "parent": parent_doc,
            "chunks": chunks_list
        }

    def _build_full_text(self, data: Dict[str, Any]) -> str:
        """Construct a single markdown document from issue parts."""
        lines = []
        lines.append(f"# [{data['key']}] {data['title']}")
        lines.append(f"**Status**: {data['status']} | **Updated**: {data['updated']}")
        lines.append(f"**URL**: {data['url']}")
        lines.append("\n## Description")
        lines.append(str(data['content']))
        
        if data['comments']:
            lines.append("\n## Comments / Discussion")
            for comment in data['comments']:
                lines.append(f"### {comment['author']} ({comment['created']})")
                lines.append(str(comment['body']))
                lines.append("---")
                
        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        """Apply filters: convert HTML/Wiki, summarize logs, replace images."""
        # A. HTML to Markdown (if applicable)
        if "bitbucket" in text or "<table>" in text or "<p>" in text:  # Heuristic detection
            text = md(text, heading_style="ATX")

        # B. Image Replacement
        # Regex for Jira Wiki style: !image.png! or [^image.png]
        text = re.sub(r'!(.+?)!', r' [IMAGE: \1 (See Jira)] ', text)
        text = re.sub(r'\[\^(.+?)\]', r' [IMAGE: \1 (See Jira)] ', text)

        # C. Log Summarization (Stack Traces)
        # Look for long blocks of "at com.example..." or similar java-like stack traces
        # Heuristic: 10+ lines starting with "at " or spaces/tabs
        
        def log_replacer(match):
            full_log = match.group(0)
            lines = full_log.split('\n')
            if len(lines) > 15:
                # Keep first 10, summarize rest
                head = "\n".join(lines[:10])
                return f"\n{head}\n... [LOG: Stack trace ({len(lines)} lines) summarized] ...\n"
            return full_log

        # Regex: blocks of lines that look like stack traces (e.g. "   at ...")
        # This is a simple approximation.
        stack_trace_pattern = r'(?:^\s+(?:at|\.\.\.) .+\n){5,}' 
        text = re.sub(stack_trace_pattern, log_replacer, text, flags=re.MULTILINE)

        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Semantic chunking:
        1. Split by Headers (##)
        2. If too large, split by Paragraphs (\n\n)
        3. Enforce max_chunk_size with overlap
        """
        chunks = []
        # split by major headers
        sections = re.split(r'(^#+ .+$)', text, flags=re.MULTILINE)
        
        current_chunk = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            # If adding this section exceeds max size, push current chunk
            if get_token_len(current_chunk) + get_token_len(section) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Helper to start new chunk with overlap (simplified: just start fresh for now)
                    current_chunk = "" 
                
                # If the section itself is huge, break it down by paragraphs
                if get_token_len(section) > self.max_chunk_size:
                    paragraphs = section.split('\n\n')
                    for p in paragraphs:
                        if get_token_len(current_chunk) + get_token_len(p) > self.max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                # Implement simple overlap by keeping bottom characters
                                overlap_chars = self.overlap * 4
                                overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                                # try to cut overlap text at the first space to avoid cutting words
                                first_space = overlap_text.find(' ')
                                if first_space != -1:
                                    overlap_text = overlap_text[first_space:]
                                current_chunk = "..." + overlap_text + "\n\n" + p
                            else:
                                current_chunk = p
                        else:
                            current_chunk += "\n\n" + p
                else:
                    current_chunk = section
            else:
                current_chunk += "\n" + section

        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
