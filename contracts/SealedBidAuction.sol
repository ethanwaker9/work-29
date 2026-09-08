// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

interface IFheCoprocessorAuction {
    function requestGeq(bytes32 a, bytes32 b) external returns (bytes32);
    function requestMux(bytes32 sel, bytes32 a, bytes32 b) external returns (bytes32);
    function requestDecrypt(bytes32 handle, address callbackTarget) external returns (uint256);
}

contract SealedBidAuction {
    IFheCoprocessorAuction public immutable fhe;
    address public immutable seller;
    uint256 public immutable closeBlock;

    bytes32 private highestBid;
    bytes32 private highestBidder;
    uint256 public bidCount;

    event BidSubmitted(address indexed bidder, bytes32 handle);
    event WinnerRevealed(uint64 bid);

    constructor(address coprocessor, uint256 blocksOpen) {
        fhe = IFheCoprocessorAuction(coprocessor);
        seller = msg.sender;
        closeBlock = block.number + blocksOpen;
    }

    function submitBid(bytes32 bidHandle) external {
        require(block.number < closeBlock, "closed");
        bytes32 better = fhe.requestGeq(bidHandle, highestBid);
        highestBid = fhe.requestMux(better, bidHandle, highestBid);
        highestBidder = fhe.requestMux(better, bytes32(uint256(uint160(msg.sender))), highestBidder);
        bidCount += 1;
        emit BidSubmitted(msg.sender, bidHandle);
    }

    function settle() external returns (uint256) {
        require(block.number >= closeBlock, "open");
        return fhe.requestDecrypt(highestBid, address(this));
    }

    function onDecryption(uint256, uint64 value) external {
        require(msg.sender == address(fhe), "only coprocessor");
        emit WinnerRevealed(value);
    }
}
