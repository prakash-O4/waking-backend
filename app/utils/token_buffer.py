class TokenBuffer:
    def __init__(self):
        self.buffer = ""
    
    def add_token(self, token: str) -> str:
        """
        Add a token to the buffer and return complete words if available.
        Returns empty string if no complete words can be formed yet.
        """
        self.buffer += token
        
        # Check if we have complete words (separated by spaces)
        if ' ' in self.buffer:
            words = self.buffer.split(' ')
            # Keep the last partial word in the buffer
            self.buffer = words[-1]
            # Return complete words joined together
            return ' '.join(words[:-1]) + ' '
        
        # If the token ends with common punctuation, we can return it
        if self.buffer.endswith(('.', ',', '!', '?', ';', ':', ')')):
            temp = self.buffer
            self.buffer = ""
            return temp
            
        return ""
