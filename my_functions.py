from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time

from pathlib import Path
from typing import Callable, Iterable, Sequence


def make_rng(seed: int = 20260920) -> random.Random:
    """
    Create a dedicated pseudo-random number generator.

    Parameters
    ----------
    seed : int
        Seed used to make the experiments reproducible.

    Returns
    -------
    random.Random
        Reproducible pseudo-random generator.
    """

    return random.Random(seed)


def system_metadata() -> dict:
    """
    Return basic hardware and software information.

    These data should be stored together with the numerical experiments
    because computational times depend on the machine and software
    environment.

    Returns
    -------
    dict
        Hardware and software information.
    """

    metadata = {
        "python_version": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "perf_counter_resolution_s": (
            time.get_clock_info("perf_counter").resolution
        ),
        "perf_counter_monotonic": (
            time.get_clock_info("perf_counter").monotonic
        ),
        "pseudo_random_generator": (
            "random.Random (Mersenne Twister; reproducible, "
            "not cryptographically secure)"
        ),
    }

    try:
        import numpy

        metadata["numpy_version"] = numpy.__version__

    except ImportError:
        metadata["numpy_version"] = None

    try:
        import matplotlib

        metadata["matplotlib_version"] = matplotlib.__version__

    except ImportError:
        metadata["matplotlib_version"] = None

    try:
        import psutil

        metadata["physical_cpus"] = psutil.cpu_count(logical=False)

        metadata["total_ram_gb"] = round(
            psutil.virtual_memory().total / (1024 ** 3),
            3,
        )

    except ImportError:

        metadata["physical_cpus"] = None
        metadata["total_ram_gb"] = None

    return metadata

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean algorithm.

    Returns integers gcd, x, and y such that

        a*x + b*y = gcd(a, b)

    Parameters
    ----------
    a, b : int
        Integer values.

    Returns
    -------
    gcd : int
        Greatest common divisor.
    x, y : int
        Bezout coefficients.
    """

    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:

        quotient = old_r // r

        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t


def modular_inverse(a: int, modulus: int) -> int:
    """
    Compute the modular inverse of a modulo modulus.

    Parameters
    ----------
    a : int
        Integer whose inverse is required.
    modulus : int
        Modulus.

    Returns
    -------
    int
        Modular inverse.

    Raises
    ------
    ValueError
        If the inverse does not exist.
    """

    gcd_value, x, _ = extended_gcd(a, modulus)

    if gcd_value != 1:
        raise ValueError("The modular inverse does not exist.")

    return x % modulus


def _decompose_power_of_two(n: int) -> tuple[int, int]:
    """
    Write n = d * 2^s, where d is odd.

    Returns
    -------
    s : int
    d : int
    """

    s = 0
    d = n

    while d % 2 == 0:

        s += 1
        d //= 2

    return s, d


def is_probable_prime(
    n: int,
    rounds: int = 40,
    rng: random.Random | None = None,
) -> bool:
    """
    Miller-Rabin primality test.

    For integers below 2^64, a deterministic set of bases is used.
    For larger integers, pseudo-random Miller-Rabin bases are used.

    Parameters
    ----------
    n : int
        Integer to test.
    rounds : int
        Number of Miller-Rabin rounds for integers >= 2^64.
    rng : random.Random, optional
        Random generator.

    Returns
    -------
    bool
        True if n is probably prime.
    """

    if n < 2:
        return False

    small_primes = (
        2, 3, 5, 7, 11, 13,
        17, 19, 23, 29, 31, 37,
    )

    if n in small_primes:
        return True

    for prime in small_primes:

        if n % prime == 0:
            return False

    s, d = _decompose_power_of_two(n - 1)

    if n < (1 << 64):

        bases = (
            2,
            325,
            9375,
            28178,
            450775,
            9780504,
            1795265022,
        )

    else:

        if rng is None:
            rng = random.Random()

        bases = [
            rng.randrange(2, n - 1)
            for _ in range(rounds)
        ]

    for base in bases:

        a = base % n

        if a in (0, 1):
            continue

        x = pow(a, d, n)

        if x in (1, n - 1):
            continue

        for _ in range(s - 1):

            x = pow(x, 2, n)

            if x == n - 1:
                break

        else:
            return False

    return True

def generate_prime(
    bits: int,
    rng: random.Random,
    rounds: int = 40,
) -> int:
    """
    Generate a probable prime with exactly the requested number of bits.

    Parameters
    ----------
    bits : int
        Prime bit length.
    rng : random.Random
        Random generator.
    rounds : int
        Miller-Rabin rounds.

    Returns
    -------
    int
        Probable prime.
    """

    if bits < 2:
        raise ValueError("'bits' must be at least 2.")

    while True:

        candidate = rng.getrandbits(bits)

        # Force the highest bit to guarantee the bit length.
        candidate |= 1 << (bits - 1)

        # Force odd parity.
        candidate |= 1

        if is_probable_prime(
            candidate,
            rounds=rounds,
            rng=rng,
        ):
            return candidate

def next_probable_prime(
    n: int,
    rng: random.Random,
    rounds: int = 40,
) -> int:
    """
    Return the first probable prime greater than or equal to n.

    Parameters
    ----------
    n : int
        Initial integer.
    rng : random.Random
        Random generator.
    rounds : int
        Miller-Rabin rounds.

    Returns
    -------
    int
        Probable prime.
    """

    if n <= 2:
        return 2

    candidate = n if n % 2 else n + 1

    while not is_probable_prime(
        candidate,
        rounds=rounds,
        rng=rng,
    ):

        candidate += 2

    return candidate

def choose_public_exponent(
    phi_n: int,
    preferred_e: int = 65537,
) -> int:
    """
    Select the RSA public exponent.

    The preferred exponent is used whenever it is coprime with phi(n).

    Parameters
    ----------
    phi_n : int
        Euler totient of n.
    preferred_e : int
        Preferred public exponent.

    Returns
    -------
    int
        Public exponent e.
    """

    if (
        1 < preferred_e < phi_n
        and math.gcd(preferred_e, phi_n) == 1
    ):
        return preferred_e

    e = 3

    while e < phi_n:

        if math.gcd(e, phi_n) == 1:
            return e

        e += 2

    raise ValueError("No valid public exponent was found.")


def rsa_keypair_from_primes(
    p: int,
    q: int,
    preferred_e: int = 65537,
) -> dict:
    """
    Build an RSA key pair from two distinct prime numbers.

    Parameters
    ----------
    p, q : int
        Distinct prime numbers.
    preferred_e : int
        Preferred public exponent.

    Returns
    -------
    dict
        RSA parameters.
    """

    if p == q:
        raise ValueError("p and q must be distinct.")

    n = p * q

    phi_n = (p - 1) * (q - 1)

    e = choose_public_exponent(
        phi_n,
        preferred_e=preferred_e,
    )

    d = modular_inverse(e, phi_n)

    return {
        "p": p,
        "q": q,
        "n": n,
        "phi_n": phi_n,
        "e": e,
        "d": d,
        "modulus_bits": n.bit_length(),
        "p_bits": p.bit_length(),
        "q_bits": q.bit_length(),
    }


def generate_rsa_keypair(
    modulus_bits: int,
    rng: random.Random,
    preferred_e: int = 65537,
    rounds: int = 40,
) -> dict:
    """
    Generate a balanced RSA key pair.

    Both prime factors have approximately half the bit length
    of the RSA modulus.

    Parameters
    ----------
    modulus_bits : int
        Requested RSA modulus bit length.
    rng : random.Random
        Random generator.
    preferred_e : int
        Preferred public exponent.
    rounds : int
        Miller-Rabin rounds.

    Returns
    -------
    dict
        RSA parameters.
    """

    if modulus_bits < 32:
        raise ValueError(
            "'modulus_bits' must be at least 32."
        )

    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits

    while True:

        p = generate_prime(
            p_bits,
            rng=rng,
            rounds=rounds,
        )

        q = generate_prime(
            q_bits,
            rng=rng,
            rounds=rounds,
        )

        if p == q:
            continue

        n = p * q

        if n.bit_length() != modulus_bits:
            continue

        phi_n = (p - 1) * (q - 1)

        e = choose_public_exponent(
            phi_n,
            preferred_e=preferred_e,
        )

        d = modular_inverse(e, phi_n)

        return {
            "p": p,
            "q": q,
            "n": n,
            "phi_n": phi_n,
            "e": e,
            "d": d,
            "modulus_bits": n.bit_length(),
            "p_bits": p.bit_length(),
            "q_bits": q.bit_length(),
        }

def rsa_encrypt_int(
    message: int,
    e: int,
    n: int,
) -> int:
    """
    Encrypt an integer using textbook RSA.

    Parameters
    ----------
    message : int
        Integer message satisfying 0 <= message < n.
    e : int
        Public exponent.
    n : int
        RSA modulus.

    Returns
    -------
    int
        Ciphertext.
    """

    if not 0 <= message < n:
        raise ValueError(
            "The message must satisfy 0 <= message < n."
        )

    return pow(message, e, n)

def rsa_decrypt_int(
    ciphertext: int,
    d: int,
    n: int,
) -> int:
    """
    Decrypt an integer using textbook RSA.

    Parameters
    ----------
    ciphertext : int
        RSA ciphertext.
    d : int
        Private exponent.
    n : int
        RSA modulus.

    Returns
    -------
    int
        Recovered integer message.
    """

    if not 0 <= ciphertext < n:
        raise ValueError(
            "The ciphertext must satisfy 0 <= ciphertext < n."
        )

    return pow(ciphertext, d, n)

def percentile(
    values: Sequence[float],
    q: float,
) -> float:
    """
    Compute a linearly interpolated percentile.

    Parameters
    ----------
    values : sequence
        Numerical values.
    q : float
        Quantile between 0 and 1.

    Returns
    -------
    float
        Percentile value.
    """

    if len(values) == 0:
        raise ValueError("'values' cannot be empty.")

    if not 0 <= q <= 1:
        raise ValueError("'q' must lie in [0, 1].")

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )

def benchmark_operation(
    func: Callable,
    *args,
    min_batch_time_s: float = 0.02,
    repeats: int = 5,
    max_loops: int = 1_000_000,
) -> dict:
    """
    Benchmark a deterministic operation using adaptive batching.

    The operation is repeated until the batch execution time is
    sufficiently large to reduce timer noise.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    *args
        Arguments passed to func.
    min_batch_time_s : float
        Minimum duration of one timing batch.
    repeats : int
        Number of timing batches.
    max_loops : int
        Maximum calls inside one batch.

    Returns
    -------
    dict
        Per-call timing statistics in nanoseconds.
    """

    if repeats < 1:
        raise ValueError(
            "'repeats' must be at least 1."
        )

    # Warm-up
    func(*args)

    loops = 1

    while True:

        start_ns = time.perf_counter_ns()

        for _ in range(loops):
            func(*args)

        elapsed_ns = (
            time.perf_counter_ns() - start_ns
        )

        if (
            elapsed_ns >= min_batch_time_s * 1e9
            or loops >= max_loops
        ):
            break

        loops *= 2

    per_call_ns = []

    for _ in range(repeats):

        start_ns = time.perf_counter_ns()

        for _ in range(loops):
            func(*args)

        elapsed_ns = (
            time.perf_counter_ns() - start_ns
        )

        per_call_ns.append(
            elapsed_ns / loops
        )

    return {
        "loops_per_batch": loops,
        "timing_repeats": repeats,
        "median_ns": statistics.median(per_call_ns),
        "q1_ns": percentile(per_call_ns, 0.25),
        "q3_ns": percentile(per_call_ns, 0.75),
        "min_ns": min(per_call_ns),
        "max_ns": max(per_call_ns),
    }

def generate_balanced_prime_pair(
    modulus_bits: int,
    rng: random.Random,
    rounds: int = 40,
) -> tuple[int, int, int]:
    """
    Generate balanced prime factors p and q.

    The resulting semiprime n = p*q has exactly the requested
    modulus bit length.

    Parameters
    ----------
    modulus_bits : int
        Requested bit length of n.
    rng : random.Random
        Random generator.
    rounds : int
        Miller-Rabin rounds.

    Returns
    -------
    p, q, n : tuple of int
    """

    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits

    while True:

        p = generate_prime(
            p_bits,
            rng=rng,
            rounds=rounds,
        )

        q = generate_prime(
            q_bits,
            rng=rng,
            rounds=rounds,
        )

        if p == q:
            continue

        n = p * q

        if n.bit_length() == modulus_bits:
            return p, q, n

def normalized_prime_gap(
    p: int,
    q: int,
    n: int | None = None,
) -> float:
    """
    Compute the normalized prime separation

        G = |p - q| / sqrt(n)

    where n = p*q.

    Parameters
    ----------
    p, q : int
        Prime factors.
    n : int, optional
        Product p*q.

    Returns
    -------
    float
        Normalized prime gap.
    """

    if n is None:
        n = p * q

    return abs(p - q) / math.sqrt(n)

def _prime_ratio_from_normalized_gap(
    normalized_gap: float,
) -> float:
    """
    Convert a target normalized gap G into an approximate q/p ratio.

    For q >= p,

        G = (q-p) / sqrt(p*q)

    and therefore, if r = q/p,

        G = (r-1)/sqrt(r).

    Parameters
    ----------
    normalized_gap : float
        Target normalized separation.

    Returns
    -------
    float
        Approximate ratio q/p.
    """

    if normalized_gap < 0:
        raise ValueError(
            "'normalized_gap' must be non-negative."
        )

    x = (
        normalized_gap
        + math.sqrt(normalized_gap**2 + 4.0)
    ) / 2.0

    return x**2

def generate_prime_pair_by_gap(
    modulus_bits: int,
    target_normalized_gap: float,
    rng: random.Random,
    rounds: int = 40,
    max_attempts: int = 10_000,
) -> tuple[int, int, int]:
    """
    Generate two primes with approximately controlled normalized gap.

    The requested quantity is

        G = |p-q| / sqrt(p*q).

    The exact G must always be computed from the returned p and q.

    This routine is intended for the smaller semiprimes used in
    the factorization experiments.

    Parameters
    ----------
    modulus_bits : int
        Bit length of n.
    target_normalized_gap : float
        Requested normalized prime gap.
    rng : random.Random
        Random generator.
    rounds : int
        Miller-Rabin rounds.
    max_attempts : int
        Maximum attempts.

    Returns
    -------
    p, q, n : tuple of int
    """

    if target_normalized_gap < 0:
        raise ValueError(
            "'target_normalized_gap' must be non-negative."
        )

    if modulus_bits % 2 != 0:
        raise ValueError(
            "Use an even modulus bit length "
            "for the controlled-gap experiment."
        )

    prime_bits = modulus_bits // 2

    upper_limit = 1 << prime_bits

    ratio = _prime_ratio_from_normalized_gap(
        target_normalized_gap
    )

    for _ in range(max_attempts):

        p = generate_prime(
            prime_bits,
            rng=rng,
            rounds=rounds,
        )

        target_q = max(
            p + 2,
            int(round(p * ratio)),
        )

        # p is odd, therefore use an odd starting point.
        if target_q % 2 == 0:
            target_q += 1

        if target_q >= upper_limit:
            continue

        q = next_probable_prime(
            target_q,
            rng=rng,
            rounds=rounds,
        )

        if q >= upper_limit:
            continue

        if q == p:
            continue

        n = p * q

        if n.bit_length() != modulus_bits:
            continue

        return p, q, n

    raise RuntimeError(
        "Could not generate a prime pair "
        "for the requested gap configuration."
    )

def _factorization_record(
    method: str,
    n: int,
    factor: int | None,
    iterations: int,
    elapsed_s: float,
    status: str,
    restarts: int = 0,
) -> dict:
    """
    Standardize the output of all factorization methods.
    """

    cofactor = None

    factor_is_valid = (
        factor is not None
        and factor not in (1, n)
        and n % factor == 0
    )

    if factor_is_valid:

        cofactor = n // factor

        factor, cofactor = sorted(
            (factor, cofactor)
        )

    elif factor is not None:
        factor = None

        if status == "success":
            status = "invalid_factor"

    success = (
        factor is not None
        and cofactor is not None
        and factor * cofactor == n
    )

    return {
        "method": method,
        "n": n,
        "success": success,
        "factor_1": factor if success else None,
        "factor_2": cofactor if success else None,
        "iterations": iterations,
        "restarts": restarts,
        "elapsed_s": elapsed_s,
        "status": (
            "success"
            if success
            else status
        ),
    }

def trial_division_factor(
    n: int,
    max_iterations: int | None = None,
    time_limit_s: float | None = None,
) -> dict:
    """
    Factor an integer using trial division.

    Parameters
    ----------
    n : int
        Integer to factor.
    max_iterations : int, optional
        Maximum number of tested divisors.
    time_limit_s : float, optional
        Maximum execution time.

    Returns
    -------
    dict
        Standardized factorization result.
    """

    start = time.perf_counter()

    if n < 2:
        raise ValueError(
            "'n' must be at least 2."
        )

    if n % 2 == 0:

        return _factorization_record(
            method="trial_division",
            n=n,
            factor=2,
            iterations=1,
            elapsed_s=time.perf_counter() - start,
            status="success",
        )

    divisor = 3
    iterations = 0

    while divisor * divisor <= n:

        iterations += 1

        if n % divisor == 0:

            return _factorization_record(
                method="trial_division",
                n=n,
                factor=divisor,
                iterations=iterations,
                elapsed_s=time.perf_counter() - start,
                status="success",
            )

        if (
            max_iterations is not None
            and iterations >= max_iterations
        ):

            return _factorization_record(
                method="trial_division",
                n=n,
                factor=None,
                iterations=iterations,
                elapsed_s=time.perf_counter() - start,
                status="max_iterations",
            )

        if (
            time_limit_s is not None
            and iterations % 1024 == 0
            and time.perf_counter() - start
            >= time_limit_s
        ):

            return _factorization_record(
                method="trial_division",
                n=n,
                factor=None,
                iterations=iterations,
                elapsed_s=time.perf_counter() - start,
                status="timeout",
            )

        divisor += 2

    return _factorization_record(
        method="trial_division",
        n=n,
        factor=None,
        iterations=iterations,
        elapsed_s=time.perf_counter() - start,
        status="no_nontrivial_factor",
    )

def fermat_factor(
    n: int,
    max_iterations: int | None = None,
    time_limit_s: float | None = None,
) -> dict:
    """
    Factor an odd integer using Fermat's factorization method.

    Parameters
    ----------
    n : int
        Integer to factor.
    max_iterations : int, optional
        Maximum number of iterations.
    time_limit_s : float, optional
        Maximum execution time.

    Returns
    -------
    dict
        Standardized factorization result.
    """

    start = time.perf_counter()

    if n < 2:
        raise ValueError(
            "'n' must be at least 2."
        )

    if n % 2 == 0:

        return _factorization_record(
            method="fermat",
            n=n,
            factor=2,
            iterations=1,
            elapsed_s=time.perf_counter() - start,
            status="success",
        )

    a = math.isqrt(n)

    if a * a < n:
        a += 1

    iterations = 0

    while True:

        b_squared = a * a - n

        b = math.isqrt(b_squared)

        iterations += 1

        if b * b == b_squared:

            factor = a - b

            if factor not in (1, n):

                return _factorization_record(
                    method="fermat",
                    n=n,
                    factor=factor,
                    iterations=iterations,
                    elapsed_s=(
                        time.perf_counter()
                        - start
                    ),
                    status="success",
                )

        if (
            max_iterations is not None
            and iterations >= max_iterations
        ):

            return _factorization_record(
                method="fermat",
                n=n,
                factor=None,
                iterations=iterations,
                elapsed_s=(
                    time.perf_counter()
                    - start
                ),
                status="max_iterations",
            )

        if (
            time_limit_s is not None
            and iterations % 1024 == 0
            and time.perf_counter() - start
            >= time_limit_s
        ):

            return _factorization_record(
                method="fermat",
                n=n,
                factor=None,
                iterations=iterations,
                elapsed_s=(
                    time.perf_counter()
                    - start
                ),
                status="timeout",
            )

        a += 1

def pollard_rho_factor(
    n: int,
    rng: random.Random,
    max_iterations: int | None = None,
    max_restarts: int = 20,
    time_limit_s: float | None = None,
) -> dict:
    """
    Factor a composite integer using Pollard's rho algorithm.

    Parameters
    ----------
    n : int
        Integer to factor.
    rng : random.Random
        Random generator.
    max_iterations : int, optional
        Maximum total number of iterations.
    max_restarts : int
        Maximum number of random restarts.
    time_limit_s : float, optional
        Maximum execution time.

    Returns
    -------
    dict
        Standardized factorization result.
    """

    start = time.perf_counter()

    if n < 2:
        raise ValueError(
            "'n' must be at least 2."
        )

    for small_factor in (2, 3, 5):

        if n % small_factor == 0:

            return _factorization_record(
                method="pollard_rho",
                n=n,
                factor=small_factor,
                iterations=1,
                elapsed_s=(
                    time.perf_counter()
                    - start
                ),
                status="success",
                restarts=0,
            )

    total_iterations = 0

    for restart in range(max_restarts):

        x = rng.randrange(2, n - 1)
        y = x

        c = rng.randrange(1, n - 1)

        factor = 1

        while factor == 1:

            x = (x * x + c) % n

            y = (y * y + c) % n
            y = (y * y + c) % n

            factor = math.gcd(
                abs(x - y),
                n,
            )

            total_iterations += 1

            if (
                max_iterations is not None
                and total_iterations
                >= max_iterations
            ):

                return _factorization_record(
                    method="pollard_rho",
                    n=n,
                    factor=None,
                    iterations=total_iterations,
                    elapsed_s=(
                        time.perf_counter()
                        - start
                    ),
                    status="max_iterations",
                    restarts=restart,
                )

            if (
                time_limit_s is not None
                and total_iterations % 256 == 0
                and time.perf_counter() - start
                >= time_limit_s
            ):

                return _factorization_record(
                    method="pollard_rho",
                    n=n,
                    factor=None,
                    iterations=total_iterations,
                    elapsed_s=(
                        time.perf_counter()
                        - start
                    ),
                    status="timeout",
                    restarts=restart,
                )

        if 1 < factor < n:

            return _factorization_record(
                method="pollard_rho",
                n=n,
                factor=factor,
                iterations=total_iterations,
                elapsed_s=(
                    time.perf_counter()
                    - start
                ),
                status="success",
                restarts=restart,
            )

    return _factorization_record(
        method="pollard_rho",
        n=n,
        factor=None,
        iterations=total_iterations,
        elapsed_s=time.perf_counter() - start,
        status="max_restarts",
        restarts=max_restarts,
    )

def factor_integer(
    n: int,
    method: str,
    rng: random.Random | None = None,
    max_iterations: int | None = None,
    time_limit_s: float | None = None,
    max_restarts: int = 20,
) -> dict:
    """
    Dispatch the factorization algorithm.

    Available methods
    -----------------
    trial_division
    fermat
    pollard_rho
    """

    method = method.lower()

    if method == "trial_division":

        return trial_division_factor(
            n,
            max_iterations=max_iterations,
            time_limit_s=time_limit_s,
        )

    if method == "fermat":

        return fermat_factor(
            n,
            max_iterations=max_iterations,
            time_limit_s=time_limit_s,
        )

    if method == "pollard_rho":

        if rng is None:
            rng = random.Random()

        return pollard_rho_factor(
            n,
            rng=rng,
            max_iterations=max_iterations,
            max_restarts=max_restarts,
            time_limit_s=time_limit_s,
        )

    raise ValueError(
        "Unknown method. Use "
        "'trial_division', "
        "'fermat', or "
        "'pollard_rho'."
    )

def run_rsa_scaling_experiment(
    modulus_bits_values: Iterable[int],
    repetitions: int = 30,
    seed: int = 20260920,
    preferred_e: int = 65537,
    rounds: int = 40,
    timing_repeats: int = 5,
    min_batch_time_s: float = 0.02,
    keep_keys: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Measure RSA key generation, encryption, and decryption costs.

    One independently generated RSA key pair is used for each
    repetition.

    Parameters
    ----------
    modulus_bits_values : iterable
        RSA modulus sizes.
    repetitions : int
        Independent repetitions per modulus size.
    seed : int
        Reproducibility seed.
    preferred_e : int
        Preferred public exponent.
    rounds : int
        Miller-Rabin rounds.
    timing_repeats : int
        Timing batches for encryption/decryption.
    min_batch_time_s : float
        Minimum duration of each adaptive timing batch.
    keep_keys : bool
        Store p, q, n, phi(n), and d in the result.
    verbose : bool
        Print progress information.

    Returns
    -------
    list of dict
        Experimental results.
    """

    if repetitions < 1:
        raise ValueError(
            "'repetitions' must be at least 1."
        )

    rng = make_rng(seed)

    records = []

    for modulus_bits in modulus_bits_values:

        if verbose:
            print(
                f"RSA experiment: "
                f"{modulus_bits} bits"
            )

        size_start = time.perf_counter()

        for repetition in range(
            1,
            repetitions + 1,
        ):

            key_start_ns = (
                time.perf_counter_ns()
            )

            key = generate_rsa_keypair(
                modulus_bits=modulus_bits,
                rng=rng,
                preferred_e=preferred_e,
                rounds=rounds,
            )

            keygen_ns = (
                time.perf_counter_ns()
                - key_start_ns
            )

            message = rng.randrange(
                2,
                key["n"] - 1,
            )

            ciphertext = rsa_encrypt_int(
                message,
                key["e"],
                key["n"],
            )

            encryption = benchmark_operation(
                rsa_encrypt_int,
                message,
                key["e"],
                key["n"],
                repeats=timing_repeats,
                min_batch_time_s=(
                    min_batch_time_s
                ),
            )

            decryption = benchmark_operation(
                rsa_decrypt_int,
                ciphertext,
                key["d"],
                key["n"],
                repeats=timing_repeats,
                min_batch_time_s=(
                    min_batch_time_s
                ),
            )

            recovered_message = (
                rsa_decrypt_int(
                    ciphertext,
                    key["d"],
                    key["n"],
                )
            )

            if recovered_message != message:

                raise RuntimeError(
                    "RSA validation failed."
                )

            record = {
                "experiment": "rsa_scaling",
                "seed": seed,
                "requested_modulus_bits": (
                    modulus_bits
                ),
                "actual_modulus_bits": (
                    key["modulus_bits"]
                ),
                "repetition": repetition,
                "e": key["e"],
                "keygen_s": (
                    keygen_ns / 1e9
                ),
                "encrypt_s": (
                    encryption["median_ns"]
                    / 1e9
                ),
                "decrypt_s": (
                    decryption["median_ns"]
                    / 1e9
                ),
                "encrypt_q1_s": (
                    encryption["q1_ns"]
                    / 1e9
                ),
                "encrypt_q3_s": (
                    encryption["q3_ns"]
                    / 1e9
                ),
                "decrypt_q1_s": (
                    decryption["q1_ns"]
                    / 1e9
                ),
                "decrypt_q3_s": (
                    decryption["q3_ns"]
                    / 1e9
                ),
                "encrypt_loops_per_batch": (
                    encryption[
                        "loops_per_batch"
                    ]
                ),
                "decrypt_loops_per_batch": (
                    decryption[
                        "loops_per_batch"
                    ]
                ),
                "message_bits": (
                    message.bit_length()
                ),
                "ciphertext_bits": (
                    ciphertext.bit_length()
                ),
                "rsa_model": "textbook_rsa",
                "padding_scheme": "none",
                "uses_crt": False,
                "message_verified": True,
            }

            if keep_keys:

                record.update(
                    {
                        "p": key["p"],
                        "q": key["q"],
                        "n": key["n"],
                        "phi_n": key["phi_n"],
                        "d": key["d"],
                    }
                )

            records.append(record)

        if verbose:

            elapsed = (
                time.perf_counter()
                - size_start
            )

            print(
                f"Completed in "
                f"{elapsed:.2f} s\n"
            )

    return records

def run_factorization_scaling_experiment(
    modulus_bits_values: Iterable[int],
    repetitions: int = 30,
    methods: Sequence[str] = (
        "trial_division",
        "fermat",
        "pollard_rho",
    ),
    seed: int = 20260920,
    rounds: int = 40,
    time_limit_s: float | None = 5.0,
    max_iterations: int | None = None,
    max_restarts: int = 20,
    verbose: bool = True,
) -> list[dict]:
    """
    Compare factorization methods as the semiprime size increases.

    Within each repetition, every factorization method receives
    exactly the same semiprime n.

    Parameters
    ----------
    modulus_bits_values : iterable
        Semiprime modulus bit lengths.
    repetitions : int
        Independent semiprimes per bit length.
    methods : sequence
        Factorization methods.
    seed : int
        Reproducibility seed.
    rounds : int
        Miller-Rabin rounds.
    time_limit_s : float, optional
        Maximum time allowed for each factorization attempt.
    max_iterations : int, optional
        Maximum iterations per algorithm.
    max_restarts : int
        Maximum Pollard-rho restarts.
    verbose : bool
        Print progress information.

    Returns
    -------
    list of dict
        Factorization results.
    """

    rng = make_rng(seed)

    records = []

    for modulus_bits in modulus_bits_values:

        if verbose:

            print(
                f"Factorization experiment: "
                f"{modulus_bits} bits"
            )

        size_start = time.perf_counter()

        for repetition in range(
            1,
            repetitions + 1,
        ):

            p, q, n = (
                generate_balanced_prime_pair(
                    modulus_bits=modulus_bits,
                    rng=rng,
                    rounds=rounds,
                )
            )

            prime_gap = abs(p - q)

            gap_normalized = (
                normalized_prime_gap(
                    p,
                    q,
                    n,
                )
            )

            for method in methods:

                method_seed = (
                    rng.getrandbits(64)
                )

                method_rng = random.Random(
                    method_seed
                )

                result = factor_integer(
                    n=n,
                    method=method,
                    rng=method_rng,
                    max_iterations=(
                        max_iterations
                    ),
                    time_limit_s=(
                        time_limit_s
                    ),
                    max_restarts=(
                        max_restarts
                    ),
                )

                record = {
                    "experiment": (
                        "factorization_scaling"
                    ),
                    "seed": seed,
                    "method_seed": method_seed,
                    "requested_modulus_bits": (
                        modulus_bits
                    ),
                    "actual_modulus_bits": (
                        n.bit_length()
                    ),
                    "repetition": repetition,
                    "p": p,
                    "q": q,
                    "n": n,
                    "prime_gap": prime_gap,
                    "normalized_gap": (
                        gap_normalized
                    ),
                }

                record.update(result)

                records.append(record)

        if verbose:

            elapsed = (
                time.perf_counter()
                - size_start
            )

            print(
                f"Completed in "
                f"{elapsed:.2f} s\n"
            )

    return records

def run_gap_experiment(
    modulus_bits_values: Iterable[int],
    target_normalized_gaps: Iterable[float],
    repetitions: int = 30,
    methods: Sequence[str] = (
        "fermat",
        "pollard_rho",
    ),
    seed: int = 20260920,
    rounds: int = 40,
    time_limit_s: float | None = 5.0,
    max_iterations: int | None = None,
    max_restarts: int = 20,
    verbose: bool = True,
) -> list[dict]:
    """
    Study the effect of prime separation on factorization cost.

    The controlled quantity is

        G = |p-q| / sqrt(p*q).

    Parameters
    ----------
    modulus_bits_values : iterable
        Semiprime modulus sizes.
    target_normalized_gaps : iterable
        Requested G values.
    repetitions : int
        Independent prime pairs per configuration.
    methods : sequence
        Factorization methods.
    seed : int
        Reproducibility seed.
    rounds : int
        Miller-Rabin rounds.
    time_limit_s : float, optional
        Maximum time per factorization attempt.
    max_iterations : int, optional
        Maximum iterations.
    max_restarts : int
        Maximum Pollard-rho restarts.
    verbose : bool
        Print progress.

    Returns
    -------
    list of dict
        Experimental records.
    """

    rng = make_rng(seed)

    records = []

    for modulus_bits in modulus_bits_values:

        for target_gap in (
            target_normalized_gaps
        ):

            if verbose:

                print(
                    f"Gap experiment: "
                    f"{modulus_bits} bits, "
                    f"G = {target_gap}"
                )

            configuration_start = (
                time.perf_counter()
            )

            for repetition in range(
                1,
                repetitions + 1,
            ):

                p, q, n = (
                    generate_prime_pair_by_gap(
                        modulus_bits=(
                            modulus_bits
                        ),
                        target_normalized_gap=(
                            target_gap
                        ),
                        rng=rng,
                        rounds=rounds,
                    )
                )

                prime_gap = abs(p - q)

                actual_gap = (
                    normalized_prime_gap(
                        p,
                        q,
                        n,
                    )
                )

                for method in methods:

                    method_seed = (
                        rng.getrandbits(64)
                    )

                    method_rng = (
                        random.Random(
                            method_seed
                        )
                    )

                    result = factor_integer(
                        n=n,
                        method=method,
                        rng=method_rng,
                        max_iterations=(
                            max_iterations
                        ),
                        time_limit_s=(
                            time_limit_s
                        ),
                        max_restarts=(
                            max_restarts
                        ),
                    )

                    record = {
                        "experiment": (
                            "prime_gap"
                        ),
                        "seed": seed,
                        "method_seed": (
                            method_seed
                        ),
                        "requested_modulus_bits": (
                            modulus_bits
                        ),
                        "actual_modulus_bits": (
                            n.bit_length()
                        ),
                        "target_normalized_gap": (
                            target_gap
                        ),
                        "actual_normalized_gap": (
                            actual_gap
                        ),
                        "repetition": (
                            repetition
                        ),
                        "p": p,
                        "q": q,
                        "n": n,
                        "prime_gap": (
                            prime_gap
                        ),
                    }

                    record.update(result)

                    records.append(record)

            if verbose:

                elapsed = (
                    time.perf_counter()
                    - configuration_start
                )

                print(
                    f"Completed in "
                    f"{elapsed:.2f} s\n"
                )

    return records

def summarize_records(
    records: Sequence[dict],
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> list[dict]:
    """
    Compute median, first quartile, third quartile, and IQR.

    Parameters
    ----------
    records : sequence of dict
        Experimental records.
    group_by : sequence of str
        Fields used to define groups.
    metrics : sequence of str
        Numerical fields to summarize.

    Returns
    -------
    list of dict
        Grouped statistics.
    """

    groups = {}

    for record in records:

        key = tuple(
            record[field]
            for field in group_by
        )

        groups.setdefault(
            key,
            [],
        ).append(record)

    summary = []

    for key, group in groups.items():

        row = {
            field: value
            for field, value
            in zip(group_by, key)
        }

        row["n_records"] = len(group)

        for metric in metrics:

            values = [
                float(record[metric])
                for record in group
                if record.get(metric)
                is not None
            ]

            if len(values) == 0:

                row[
                    f"{metric}_median"
                ] = None

                row[
                    f"{metric}_q1"
                ] = None

                row[
                    f"{metric}_q3"
                ] = None

                row[
                    f"{metric}_iqr"
                ] = None

                continue

            q1 = percentile(
                values,
                0.25,
            )

            median = percentile(
                values,
                0.50,
            )

            q3 = percentile(
                values,
                0.75,
            )

            row[
                f"{metric}_median"
            ] = median

            row[
                f"{metric}_q1"
            ] = q1

            row[
                f"{metric}_q3"
            ] = q3

            row[
                f"{metric}_iqr"
            ] = q3 - q1

        summary.append(row)

    return summary

def summarize_factorization(
    records: Sequence[dict],
    group_by: Sequence[str],
) -> list[dict]:
    """
    Summarize factorization performance.

    The function reports success rate and statistics calculated
    only from successful factorizations.

    Parameters
    ----------
    records : sequence of dict
        Factorization records.
    group_by : sequence of str
        Fields defining each experimental group.

    Returns
    -------
    list of dict
        Factorization summary.
    """

    groups = {}

    for record in records:

        key = tuple(
            record[field]
            for field in group_by
        )

        groups.setdefault(
            key,
            [],
        ).append(record)

    output = []

    for key, group in groups.items():

        row = {
            field: value
            for field, value
            in zip(group_by, key)
        }

        total = len(group)

        successful = [
            record
            for record in group
            if record.get(
                "success",
                False,
            )
        ]

        timeout_count = sum(
            record.get("status")
            == "timeout"
            for record in group
        )

        row["n_total"] = total

        row["n_success"] = len(
            successful
        )

        row["success_rate"] = (
            len(successful) / total
            if total > 0
            else None
        )

        row["n_timeout"] = (
            timeout_count
        )

        if successful:

            elapsed = [
                record["elapsed_s"]
                for record in successful
            ]

            iterations = [
                record["iterations"]
                for record in successful
            ]

            row[
                "elapsed_s_median"
            ] = percentile(
                elapsed,
                0.50,
            )

            row[
                "elapsed_s_q1"
            ] = percentile(
                elapsed,
                0.25,
            )

            row[
                "elapsed_s_q3"
            ] = percentile(
                elapsed,
                0.75,
            )

            row[
                "iterations_median"
            ] = percentile(
                iterations,
                0.50,
            )

            row[
                "iterations_q1"
            ] = percentile(
                iterations,
                0.25,
            )

            row[
                "iterations_q3"
            ] = percentile(
                iterations,
                0.75,
            )

        else:

            row[
                "elapsed_s_median"
            ] = None

            row[
                "elapsed_s_q1"
            ] = None

            row[
                "elapsed_s_q3"
            ] = None

            row[
                "iterations_median"
            ] = None

            row[
                "iterations_q1"
            ] = None

            row[
                "iterations_q3"
            ] = None

        output.append(row)

    return output

def save_records_csv(
    records: Sequence[dict],
    filename: str | Path,
) -> Path:
    """
    Save experimental records to CSV.

    Parameters
    ----------
    records : sequence of dict
        Experimental data.
    filename : str or Path
        Output filename.

    Returns
    -------
    Path
        Saved file path.
    """

    if len(records) == 0:
        raise ValueError(
            "'records' cannot be empty."
        )

    path = Path(filename)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []
    seen = set()

    for record in records:

        for key in record:

            if key not in seen:

                seen.add(key)

                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(records)

    return path

def save_json(
    data,
    filename: str | Path,
) -> Path:
    """
    Save JSON-serializable information.

    Parameters
    ----------
    data
        Data to save.
    filename : str or Path
        Output filename.

    Returns
    -------
    Path
        Saved file path.
    """

    path = Path(filename)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path

def validate_rsa_implementation() -> bool:
    """
    Validate the RSA implementation using a small deterministic case.

    Returns
    -------
    bool
        True if all messages are correctly recovered.
    """

    key = rsa_keypair_from_primes(
        p=61,
        q=53,
        preferred_e=17,
    )

    test_messages = (
        2,
        42,
        65,
        123,
        1024,
    )

    for message in test_messages:

        ciphertext = rsa_encrypt_int(
            message,
            key["e"],
            key["n"],
        )

        recovered = rsa_decrypt_int(
            ciphertext,
            key["d"],
            key["n"],
        )

        if recovered != message:
            return False

    return True

def validate_factorization_implementations(
    seed: int = 12345,
) -> dict:
    """
    Validate all factorization methods on the same semiprime.

    Returns
    -------
    dict
        Validation status for each algorithm.
    """

    p = 83
    q = 97

    n = p * q

    rng = make_rng(seed)

    results = {
        "trial_division":
            trial_division_factor(n),

        "fermat":
            fermat_factor(n),

        "pollard_rho":
            pollard_rho_factor(
                n,
                rng=rng,
            ),
    }

    validation = {}

    for method, result in results.items():

        validation[method] = (
            result["success"]
            and
            result["factor_1"]
            * result["factor_2"]
            == n
        )

    return validation