from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from web3 import Web3
from web3.contract import Contract

from config.settings import SETTINGS
from utils.logging import get_logger

load_dotenv()

logger = get_logger("blockchain")


class BlockchainDisabled(Exception):
    """Raised when blockchain anchoring is disabled in settings."""


class BlockchainConfigurationError(Exception):
    """Raised when blockchain anchoring is enabled but misconfigured."""


@dataclass(frozen=True)
class _Client:
    w3: Web3
    account_address: str
    private_key: str
    contract: Contract
    gas_limit: int
    wait_for_receipt: bool

    def anchor(self, txid: str) -> Dict[str, Any]:
        if len(txid) != 64:
            raise ValueError("txid must be a 64-character hex string")

        tx_bytes = bytes.fromhex(txid)
        nonce = self.w3.eth.get_transaction_count(self.account_address)
        base_tx = {
            "from": self.account_address,
            "nonce": nonce,
            "gas": self.gas_limit,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.w3.eth.chain_id,
        }

        tx = self.contract.functions.anchorHash(tx_bytes).build_transaction(base_tx)
        signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

        if not self.wait_for_receipt:
            return {
                "status": "submitted",
                "tx_hash": tx_hash.hex(),
                "contract_address": self.contract.address,
                "sender": self.account_address,
            }

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return {
            "status": "anchored",
            "tx_hash": tx_hash.hex(),
            "block_number": receipt.blockNumber,
            "contract_address": self.contract.address,
            "sender": self.account_address,
        }


def anchor_txid(txid: str, enabled: Optional[bool] = None) -> Dict[str, Any]:
    """
    Anchors the provided txid on-chain when blockchain support is enabled.

    Returns a dict with at least a `status` field. Possible values:
      - "disabled": feature turned off.
      - "error": configuration or runtime failure (see `error` key).
      - "submitted"/"anchored": successful submission, possibly with receipt info.
    """
    cfg = SETTINGS.blockchain
    enabled_flag = cfg.enabled if enabled is None else bool(enabled)
    if not enabled_flag:
        return {"status": "disabled"}

    try:
        client = _get_client()
    except BlockchainDisabled:
        return {"status": "disabled"}
    except BlockchainConfigurationError as exc:
        logger.warning("blockchain:configuration_error", extra={"extra_fields": {"error": str(exc)}})
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("blockchain:unexpected_init_error")
        return {"status": "error", "error": str(exc)}

    try:
        result = client.anchor(txid)
        logger.info("blockchain:anchor_ok", extra={"extra_fields": {"tx_hash": result.get("tx_hash"), "status": result.get("status")}})
        return result
    except Exception as exc:
        logger.warning("blockchain:anchor_failed", extra={"extra_fields": {"error": str(exc)}})
        return {"status": "error", "error": str(exc)}


@lru_cache(maxsize=1)
def _get_client() -> _Client:
    cfg = SETTINGS.blockchain
    if not cfg.enabled:
        raise BlockchainDisabled()

    contract_path = Path(cfg.contract_info_path)
    if not contract_path.exists():
        raise BlockchainConfigurationError(f"contract_info.json not found at {contract_path}")

    rpc_url = os.getenv(cfg.rpc_url_env)
    if not rpc_url:
        raise BlockchainConfigurationError(f"Environment variable {cfg.rpc_url_env} is required")

    private_key = os.getenv(cfg.private_key_env)
    if not private_key:
        raise BlockchainConfigurationError(f"Environment variable {cfg.private_key_env} is required")

    with contract_path.open("r", encoding="utf-8") as fh:
        info = json.load(fh)

    address = info.get("address")
    abi = info.get("abi")
    if not address or not abi:
        raise BlockchainConfigurationError("contract_info.json must include 'address' and 'abi'")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise BlockchainConfigurationError(f"Unable to connect to RPC endpoint {rpc_url}")

    checksum_address = Web3.to_checksum_address(address)
    contract = w3.eth.contract(address=checksum_address, abi=abi)
    account = w3.eth.account.from_key(private_key)

    return _Client(
        w3=w3,
        account_address=account.address,
        private_key=private_key,
        contract=contract,
        gas_limit=int(cfg.gas_limit),
        wait_for_receipt=bool(cfg.wait_for_receipt),
    )

