import zlib
from collections import Counter
from typing import List, Tuple
from shared.repositories.custom_tokenizer import tokenize

class SparseEmbedder:
    """
    Generates Sparse Vectors (indices and term frequencies) 
    using the SOC custom tokenizer and CRC32 hashing.
    """
    
    @staticmethod
    def generate_sparse_vectors(texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Tokenizes the input texts and produces SparseVector formats 
        required by Qdrant (indices and values).
        Using CRC32 to map string tokens to standard uint32 integers.
        Returns:
            sparse_indices (List of List of int)
            sparse_values (List of List of float - representing Term Frequency)
        """
        all_indices = []
        all_values = []
        
        for text in texts:
            tokens = tokenize(text)
            # Calculate Term Frequency (TF)
            tf_counter = Counter(tokens)
            
            indices = []
            values = []
            
            for token, tf in tf_counter.items():
                # Stable hashing to uint32 space
                # Qdrant accepts any non-negative integer as feature index
                idx = zlib.crc32(token.encode('utf-8'))
                indices.append(idx)
                values.append(float(tf))
                
            all_indices.append(indices)
            all_values.append(values)
            
        return all_indices, all_values

sparse_embedder = SparseEmbedder()
