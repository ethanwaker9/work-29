// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

interface IFheCoprocessor {
    function requestGeq(bytes32 a, bytes32 b) external returns (bytes32);
    function requestMux(bytes32 sel, bytes32 a, bytes32 b) external returns (bytes32);
    function requestAdd(bytes32 a, bytes32 b) external returns (bytes32);
    function requestSub(bytes32 a, bytes32 b) external returns (bytes32);
    function requestDecrypt(bytes32 handle, address callbackTarget) external returns (uint256);
}

contract ConfidentialERC20 {
    IFheCoprocessor public immutable fhe;
    mapping(address => bytes32) private balanceHandle;
    mapping(uint256 => address) private pendingDecrypt;

    event EncryptedTransfer(address indexed from, address indexed to, bytes32 amountHandle);
    event DecryptionRequested(uint256 indexed requestId, address indexed account);
    event BalanceRevealed(address indexed account, uint64 value);

    constructor(address coprocessor) {
        fhe = IFheCoprocessor(coprocessor);
    }

    function balanceOfHandle(address who) external view returns (bytes32) {
        return balanceHandle[who];
    }

    function mint(address to, bytes32 amountHandle) external {
        balanceHandle[to] = fhe.requestAdd(balanceHandle[to], amountHandle);
    }

    function transfer(address to, bytes32 amountHandle) external returns (bool) {
        bytes32 from = balanceHandle[msg.sender];
        bytes32 ok = fhe.requestGeq(from, amountHandle);
        bytes32 zero = bytes32(0);
        bytes32 amount = fhe.requestMux(ok, amountHandle, zero);
        balanceHandle[msg.sender] = fhe.requestSub(from, amount);
        balanceHandle[to] = fhe.requestAdd(balanceHandle[to], amount);
        emit EncryptedTransfer(msg.sender, to, amount);
        return true;
    }

    function requestBalanceReveal() external returns (uint256 requestId) {
        requestId = fhe.requestDecrypt(balanceHandle[msg.sender], address(this));
        pendingDecrypt[requestId] = msg.sender;
        emit DecryptionRequested(requestId, msg.sender);
    }

    function onDecryption(uint256 requestId, uint64 value) external {
        require(msg.sender == address(fhe), "only coprocessor");
        address who = pendingDecrypt[requestId];
        require(who != address(0), "unknown request");
        delete pendingDecrypt[requestId];
        emit BalanceRevealed(who, value);
    }
}
