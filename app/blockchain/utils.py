import re
import base58

chainList = [
    {
        "name": "Ethereum/Arbitrum",
        "prefix": "0x",
        "length": 42,
        "pattern": r"^0x[a-fA-F0-9]{40}$",
        "decoder": None
    },
    {
        "name": "Solana",
        "prefix": "",
        "length": 32,
        "pattern": None,
        "decoder": base58.b58decode
    }
]

def checkPubKeyAddress(address):
    for chain in chainList:
        if chain["pattern"]:
            if address.startswith(chain["prefix"]) and len(address) == chain["length"]:
                if re.match(chain["pattern"], address):
                    return {
                        "status": "valid",
                        "chain_type": chain["name"]
                    }
        else:
            try:
                decoded = chain["decoder"](address)
                if len(decoded) == chain["length"]:
                    return {
                        "status": "valid",
                        "chain_type": chain["name"]
                    }
            except ValueError:
                pass

    return {
        "status": "error",
        "message": "Invalid or unsupported chain address"
    }