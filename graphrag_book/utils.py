

def chunk_text(text: str, chunk_size: int, overlap: int, split_on_whitespaces: bool = True) -> list[str]:
    """
    Chunk text into chunks of a given size, with an overlap.
    """
    chunks = []
    index = 0
    while index < len(text):
        if split_on_whitespaces:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index = end
    return chunks
