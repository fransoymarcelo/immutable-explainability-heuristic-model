// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ExplanationNotary
 * @dev Este contrato actúa como un notario inmutable.
 * Permite a cualquiera anclar un hash de 32 bytes (ej. SHA-256)
 * emitiendo un evento que se sella en la blockchain.
 * Esto valida el concepto de "Explicabilidad Inmutable".
 */
contract ExplanationNotary {

    /**
     * @dev Evento que se emite cuando un hash es anclado.
     * @param sender La dirección que ancló el hash.
     * @param explanationHash El hash SHA-256 de la traza de auditoría (txid).
     * @param blockTimestamp La marca de tiempo del bloque.
     */
    event HashAnchored(
        address indexed sender,
        bytes32 indexed explanationHash,
        uint256 blockTimestamp
    );

    /**
     * @notice Ancla un hash de 32 bytes a la blockchain.
     * @param _hash El hash (txid) a ser anclado.
     */
    function anchorHash(bytes32 _hash) external {
        // Emite el evento. Esta es la acción clave.
        // Cuesta gas, pero sella los datos para siempre.
        emit HashAnchored(msg.sender, _hash, block.timestamp);
    }
}