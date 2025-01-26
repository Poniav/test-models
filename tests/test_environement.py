import os
import unittest
from dotenv import load_dotenv

class TestEnvironmentVariables(unittest.TestCase):

    def setUp(self):
        load_dotenv()

    def test_required_environment_variables(self):
        required_vars = [
            "DATABASE_URL",
            "SECRET_KEY",
            "FLASK_ENV"
        ]
        
        for var in required_vars:
            with self.subTest(var=var):
                self.assertIn(var, os.environ, f"Missing required environment variable: {var}")

if __name__ == '__main__':
    unittest.main()