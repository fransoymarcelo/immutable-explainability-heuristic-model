import json
import hashlib
import os
from dotenv import load_dotenv
from web3 import Web3
from solcx import compile_source, install_solc

# --- Initial configuration ---
load_dotenv()
install_solc(version='0.8.20')  # Ensure the compiler is available

# Load configuration from .env
RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not RPC_URL or not PRIVATE_KEY:
    raise EnvironmentError("RPC_URL and PRIVATE_KEY must be defined in .env")

# Connect to the Sepolia network
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise ConnectionError(f"Unable to connect to {RPC_URL}")

# Load local wallet
account = w3.eth.account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address
print(f"Connected to Sepolia as: {account.address}")


def compile_and_deploy():
    """Compile and deploy the ExplanationNotary.sol contract."""
    print("Compiling contract...")
    with open("ExplanationNotary.sol", "r") as f:
        contract_source = f.read()

    compiled_sol = compile_source(
        contract_source,
        output_values=["abi", "bin"],
        solc_version="0.8.20"
    )
    contract_id, contract_interface = compiled_sol.popitem()
    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]

    print("Deploying contract...")
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Build deployment transaction
    construct_txn = Contract.constructor().build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 2000000,
        'gasPrice': w3.eth.gas_price
    })

    signed_txn = w3.eth.account.sign_transaction(construct_txn, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    print(f"Waiting for deployment receipt (tx: {tx_hash.hex()})...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    contract_address = tx_receipt.contractAddress
    print(f"Contract deployed! Address: {contract_address}")
    
    # Persist address and ABI for later use
    with open("contract_info.json", "w") as f:
        json.dump({
            "address": contract_address,
            "abi": abi
        }, f)
    
    return contract_address, abi

def get_contract_instance():
    """Load the deployed contract instance."""
    if not os.path.exists("contract_info.json"):
        print("contract_info.json not found. Deploying a new contract...")
        return compile_and_deploy()
    
    with open("contract_info.json", "r") as f:
        info = json.load(f)
    
    return info["address"], info["abi"]

def calculate_txid(fusion_details: dict) -> str:
    """
    Deterministically compute the SHA-256 hash [cite: 8, 138].
    The JSON must be canonical (sorted keys, no extraneous whitespace).
    """
    # separators=(',', ':') produces the most compact JSON
    # sort_keys=True guarantees deterministic ordering
    canonical_json = json.dumps(
        fusion_details, 
        separators=(',', ':'), 
        sort_keys=True
    ).encode('utf-8')
    
    txid = hashlib.sha256(canonical_json).hexdigest()
    return txid

def anchor_txid(txid_hex: str, contract_address: str, abi: list):
    """
    Call the contract's anchorHash method to seal the txid.
    """
    print(f"Anchoring txid: {txid_hex}...")
    
    # Convert the hex string to bytes32
    txid_bytes = bytes.fromhex(txid_hex)
    
    # Load contract instance
    contract = w3.eth.contract(address=contract_address, abi=abi)
    
    # Build the transaction
    tx = contract.functions.anchorHash(txid_bytes).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 100000,
        'gasPrice': w3.eth.gas_price
    })

    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print(f"Waiting for anchoring receipt (tx: {tx_hash.hex()})...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print(f"Anchoring successful! txid sealed in block: {receipt.blockNumber}")
    return tx_hash.hex(), receipt.blockNumber

# --- Main execution flow ---
if __name__ == "__main__":
    
    # 1. Load or deploy contract
    CONTRACT_ADDRESS, ABI = get_contract_instance()
    print(f"Using contract at: {CONTRACT_ADDRESS}")

    # 2. Simulate a 'fusion_details' payload (placeholder for pipeline output)
    # [cite: 131-137]
    sample_fusion_details = {
        "w_text": 0.5843,
        "details": {
            "inputs": { "asr_conf": 0.958, "arousal": 0.12, "valence": 0.02 },
            "fired_rules": [
                {"if": ["asr_conf is high"], "then": "w_text is high", "strength": 0.916},
                {"if": ["valence is neu"], "then": "w_text is mid", "strength": 1.0}
            ],
            "out_sets": { "low": 0.0, "mid": 1.0, "high": 0.916 }
        }
    }
    
    # 3. Compute the txid (SHA-256 hash)
    txid = calculate_txid(sample_fusion_details)
    print(f"Computed txid (SHA-256): {txid}")
    
    # 4. Anchor the txid on-chain
    blockchain_tx, block_num = anchor_txid(txid, CONTRACT_ADDRESS, ABI)
    
    print("\n--- PoC Completed ---")
    print(f"   Payload (JSON): {sample_fusion_details}")
    print(f"   Hash (txid): {txid}")
    print(f"On-chain transaction: https://sepolia.etherscan.io/tx/0x{blockchain_tx}")
    print(f"              Block: {block_num}")