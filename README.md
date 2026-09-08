# More Efficient FHE Bootstrapping for Confidential Smart Contracts

The repository contains an FHEW/TFHE style bootstrapping stack, six blind rotation algorithms implemented on top of one shared
ring layer, the validated failure probability analysis, the NTRU asymmetric variance estimator, and a confidential smart contract workload with the accompanying Solidity contracts.

## Files and Contents

`fhe/` core library (`ring.py` negacyclic NTT over a word size prime, gadget decomposition, samplers; `counters.py` instrumentation of ring multiplications and transforms; `rlwe.py` RLWE, RLWE', RGSW, external product, automorphism key switching; `ngs.py` NTRU scalar and vector ciphertexts, external product; `walk.py` discrete logarithm tables, modulus switching, hop scheduling; `blindrot.py` AP, GINX, LMKCDEY, FINAL, XZD and the fused blind rotation; `lwe.py` LWE key switching, modulus switching, coefficient extraction; `bootstrap.py` complete gate bootstrapping pipeline; `params.py` parameter sets) · `analysis/` (`failure.py` validated failure probability (Theorem 4.3) and cost models; `optimize.py` minimal gadget for a target failure probability (Algorithm 1); `ntru_fatigue.py` balancing map, fatigue point transport, admissible modulus; `lwe_estimator.py` primal uSVP and dual core-SVP estimates) · `contracts/` (`circuits.py` Boolean circuit layer (adders, comparator, multiplexer); `gas.py` EVM gas accounting for the two on chain ciphertext designs; `ConfidentialERC20.sol` confidential token contract; `SealedBidAuction.sol` sealed bid auction contract) · `experiments/` (`e1_blindrot.py` gate bootstrapping microbenchmark for the six methods; `e2_tradeoff.py` window and fusion depth sweep; `e3_contract.py` confidential transfer workload) · `figures/` (`make_figures.py` all figures, written as EPS; `tests/test_rotation.py` correctness of all six blind rotations against a reference rotation · `results/` JSON produced by the experiments


## Running the Experiments
This requires Python 3.10 or later with `numpy`, `scipy` and `matplotlib` with `epstopdf`, which ships with TeX Live and MacTeX.
```
python3 tests/test_rotation.py          # correctness of all six blind rotations
python3 experiments/e1_blindrot.py      
python3 experiments/e2_tradeoff.py      
python3 analysis/optimize.py            # minimal gadget dimension per method
python3 experiments/e3_contract.py      
python3 figures/make_figures.py         
python3 figures/make_tables.py          
```
or simply
```
python3 run_all.py
```

The whole pipeline runs in well under two hours on one core of a laptop. The microbenchmark takes about nine minutes, the trade-off sweep about two minutes and the contract workload about twenty five minutes; the analysis scripts are instantaneous.
```
python3 analysis/failure.py             # validated against heuristic failure bound
python3 analysis/ntru_fatigue.py        # invariance check and admissible modulus
python3 analysis/lwe_estimator.py       # security estimates, with calibration cases
```
`analysis/lwe_estimator.py` reproduces the published core-SVP hardness of Kyber-512 and of the Homomorphic Encryption Standard entries as calibration.

## Parameters
The three parameter sets of our research are in `fhe/params.py`. All of them use ring
dimension `N = 1024`, ring modulus slightly below `2^27`, gadget `(d, B) = (3, 2^8)`,
key switching modulus `2^18` with gadget `(5, 2^4)`, accumulator noise `sigma_g = 3.2`
and NTRU secret `sigma_f = 32`. They differ only in the distribution of the LWE secret
key: ternary at `n = 512`, Gaussian of variance `4/3` at `n = 465`, and Gaussian of
variance `16/3` at `n = 465`.

## Smart contract layer
`contracts/ConfidentialERC20.sol` and `contracts/SealedBidAuction.sol` are the on-chain
side of the workload. They hold a 32-byte handle per encrypted value, delegate every
homomorphic operation to a coprocessor interface, and receive revealed plaintexts
through a callback, which is the interface a threshold decryption committee exposes.
`contracts/gas.py` accounts for the calldata and storage cost of the two designs, one
that keeps whole ciphertexts on chain and one that keeps handles. The Python driver in
`experiments/e3_contract.py` executes the transfer logic of the token contract under
the FHE stack and checks the decrypted post state.
