import unittest
from app.utils import checkPubKeyAddress

class TestAddressValidation(unittest.TestCase):
    
    def test_ethereum_address(self):
        result = checkPubKeyAddress("0x32Be343B94f860124dC4fEe278FDCBD38C102D88")
        self.assertEqual(result['status'], "valid")
        self.assertEqual(result['chain_type'], "Ethereum/Arbitrum")

    def test_solana_address(self):
        result = checkPubKeyAddress("3N5U3Xh8huwFD8xNzYrZG5yLs6K2V9nb3r6oQ9A74kGz")
        self.assertEqual(result['status'], "valid")
        self.assertEqual(result['chain_type'], "Solana")

    def test_empty_address(self):
        result = checkPubKeyAddress("")
        self.assertEqual(result['status'], "error")
        self.assertEqual(result['message'], "Invalid or unsupported chain address")
    
    def test_invalid_address(self):
        result = checkPubKeyAddress("invalid_address")
        self.assertEqual(result['status'], "error")
        self.assertEqual(result['message'], "Invalid or unsupported chain address")


if __name__ == '__main__':
    unittest.main()