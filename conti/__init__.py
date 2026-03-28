"""conti-safety"""
__version__ = "0.1.0"

try:
    from dotenv import load_dotenv
    load_dotenv()  # auto-load keys from .env file
except ImportError:
    pass
