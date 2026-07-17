import os
import sys
import json  # unused - will trigger F401

def add_numbers(a, b):
    result = a + b
    unused_var = 42  # unused - will trigger F841
    return result

def format_message(msg):
    return f"Message: {msg}"

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        results = []
        for item in self.data:
            results.append(item * 2)
        return results
