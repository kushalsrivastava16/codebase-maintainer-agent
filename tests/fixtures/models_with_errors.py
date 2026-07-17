import os
import sys
import json  # unused - F401
import re    # unused - F401
from pathlib import Path  # unused - F401

VERY_LONG_CONSTANT = "this string is intentionally very long and exceeds the line length limit set in pyproject toml which is 99 characters"

def get_user(user_id, database_connection, include_metadata=False, include_permissions=False, include_history=False):
    unused_result = database_connection.query(f"SELECT * FROM users WHERE id = {user_id}")
    return unused_result

class UserModel:
    def __init__(self, name, email, age, role, department, location, created_at, updated_at):
        self.name = name
        self.email = email
        self.age = age
        self.role = role
        self.department = department
        self.location = location
        self.created_at = created_at
        self.updated_at = updated_at
        x = 99  # unused variable - F841

    def validate(self):
        errors = []
        if not self.name:
            errors.append("name is required")
        if not self.email:
            errors.append("email is required")
        if self.age is not None and not isinstance(self.age, int):
            errors.append("age must be an integer")
        return errors

    def to_dict(self):
        return {"name": self.name, "email": self.email, "age": self.age, "role": self.role, "department": self.department}
