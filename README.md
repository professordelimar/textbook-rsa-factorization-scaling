# Textbook RSA and Integer Factorization Scaling

This repository contains the reproducible computational material associated
with the study *Computational Scaling of Textbook RSA and the Influence of
Prime-Factor Separation on Classical Integer Factorization*.

The experiments investigate the computational scaling of textbook RSA key
generation, encryption, and decryption, as well as the performance of three
classical integer-factorization methods: Trial Division, Fermat Factorization,
and Pollard's Rho. The notebook also evaluates how normalized prime-factor
separation affects factorization cost.

## Repository contents

- `rsa_code.ipynb`: executed Jupyter notebook containing the numerical
  experiments, validation checks, tables, figures, timing measurements, and
  computational-environment report.
- `my_functions.py`: pure-Python implementation of the RSA operations,
  probable-prime generation, factorization algorithms, experiment drivers,
  validation routines, summaries, and data-export functions.

## Computational experiments

The notebook contains three principal experiments:

1. computational scaling of textbook RSA key generation, encryption, and
   decryption as the modulus size increases;
2. scaling comparison among Trial Division, Fermat Factorization, and
   Pollard's Rho using the same semiprime instances;
3. analysis of the relationship between normalized prime-factor separation
   and factorization cost.

The normalized prime-factor separation is defined as

```text
G = |p - q| / sqrt(pq).
```

## Requirements

- Python 3.10 or later
- Jupyter Notebook or JupyterLab
- NumPy
- Matplotlib
- psutil, optional but recommended for the hardware and memory report

Install the required packages with:

```bash
python -m pip install jupyter numpy matplotlib psutil
```

## Reproducing the experiments

Keep `rsa_code.ipynb` and `my_functions.py` in the same directory. Then start
Jupyter:

```bash
jupyter notebook rsa_code.ipynb
```

Run the notebook cells in order. The pseudo-random generator uses a fixed seed
to support reproducibility. The notebook records experiment durations and the
available hardware and software configuration without rerunning completed
benchmarks in its final reporting cell.

Generated numerical tables, timing records, fitted summaries, and publication
figures are written to the notebook directory.

## Scope and security notice

The implementation uses textbook RSA with integer messages, no padding, and
no Chinese Remainder Theorem optimization. It is intended exclusively for
scientific and educational analysis. The deterministic pseudo-random generator
and textbook RSA implementation must not be used in production cryptographic
systems.

The reported timings characterize this pure-Python implementation and should
not be interpreted as benchmarks of TLS, OpenSSL, hardware-accelerated
cryptography, or commercial cryptographic software.

## Authors

- Nelson Ribeiro-Filho, Instituto Federal de Educacao, Ciencia e Tecnologia do
  Para (IFPA)
- Wendel Melo Prudencio de Araujo, Instituto Federal de Educacao, Ciencia e
  Tecnologia do Para (IFPA)
- Tarcyana do Socorro Figueiredo de Sousa, Universidade Federal Rural da
  Amazonia (UFRA)
- Leandro Santana da Costa Filho, Universidade Federal Rural da Amazonia
  (UFRA)

## Citation

If you use this repository, please cite the associated scientific article.
Complete bibliographic information and the article DOI will be added after
publication.

