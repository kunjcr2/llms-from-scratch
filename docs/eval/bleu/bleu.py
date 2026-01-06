"""
Minimal BLEU scorer (no external libs). Supports multiple references.
Computes BLEU-1 through BLEU-4 with uniform weights.
"""

import math


def get_ngrams(tokens, n):
    """
    Return dict of n-gram counts for a token list.
    
    Args:
        tokens: list of words
        n: n-gram order (1 for unigrams, 2 for bigrams, etc.)
    Returns:
        dict mapping n-gram tuples to their counts
    """
    counts = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def clipped_precision(candidate, references, n):
    """
    Compute clipped n-gram precision across multiple references.
    
    Clips candidate n-gram counts to the max count found in any reference,
    preventing inflated scores from repeated words.
    
    Args:
        candidate: list of tokens
        references: list of token lists
        n: n-gram order
    Returns:
        precision score (float 0-1)
    """
    cand_ngrams = get_ngrams(candidate, n)
    
    # Build max count per n-gram across all references
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        for ng, cnt in ref_ngrams.items():
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), cnt)
    
    # Clip candidate counts to reference max
    clipped = sum(
        min(cnt, max_ref_counts.get(ng, 0)) 
        for ng, cnt in cand_ngrams.items()
    )
    total = sum(cand_ngrams.values())
    
    return clipped / total if total > 0 else 0.0


def brevity_penalty(candidate, references):
    """
    Compute brevity penalty using closest reference length.
    
    Penalizes candidates shorter than the closest reference length.
    BP = 1 if c >= r, else BP = e^(1 - r/c)
    
    Args:
        candidate: list of tokens
        references: list of token lists
    Returns:
        brevity penalty (float 0-1)
    """
    c = len(candidate)
    ref_lens = [len(r) for r in references]
    
    # Pick reference length closest to candidate (prefer shorter if tie)
    r = min(ref_lens, key=lambda rl: (abs(rl - c), rl))
    
    if c >= r:
        return 1.0
    return math.exp(1 - r / c) if c > 0 else 0.0


def bleu(candidate, references, max_n=4, smoothing=True):
    """
    Compute BLEU score with uniform weights up to max_n.
    
    BLEU = BP * exp(1/N * sum(log(p_n)))
    
    Args:
        candidate: string (tokenized by whitespace, lowercased)
        references: list of reference strings
        max_n: max n-gram order (default 4)
        smoothing: if True, add epsilon smoothing for zero counts
    Returns:
        BLEU score (float 0-1)
    """
    # Tokenize: lowercase and split on whitespace
    cand_tokens = candidate.lower().split()
    ref_tokens = [r.lower().split() for r in references]
    
    # Limit max_n to candidate length (can't have 4-grams with 3 words)
    effective_n = min(max_n, len(cand_tokens))
    if effective_n == 0:
        return 0.0
    
    # Collect log precisions for geometric mean
    log_prec_sum = 0.0
    epsilon = 0.1  # Smoothing value for zero precisions
    
    for n in range(1, effective_n + 1):
        p = clipped_precision(cand_tokens, ref_tokens, n)
        if p == 0:
            if smoothing:
                # Add-epsilon smoothing: treat as small non-zero value
                p = epsilon / (len(cand_tokens) - n + 1)
            else:
                return 0.0  # Any zero precision → BLEU = 0
        log_prec_sum += math.log(p)
    
    # Geometric mean of precisions
    geo_mean = math.exp(log_prec_sum / effective_n)
    
    # Apply brevity penalty
    bp = brevity_penalty(cand_tokens, ref_tokens)
    
    return bp * geo_mean


def bleu_ngram_details(candidate, references, max_n=4, smoothing=True):
    """
    Compute BLEU score and return individual n-gram precisions.
    
    Returns:
        dict with keys: 'bleu', 'bp', '1-gram', '2-gram', '3-gram', '4-gram'
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = [r.lower().split() for r in references]
    
    effective_n = min(max_n, len(cand_tokens))
    if effective_n == 0:
        return {'bleu': 0.0, 'bp': 0.0, '1-gram': 0.0, '2-gram': 0.0, '3-gram': 0.0, '4-gram': 0.0}
    
    precisions = {}
    log_prec_sum = 0.0
    epsilon = 0.1
    
    for n in range(1, max_n + 1):
        if n <= effective_n:
            p = clipped_precision(cand_tokens, ref_tokens, n)
            if p == 0 and smoothing:
                p = epsilon / (len(cand_tokens) - n + 1)
            precisions[f'{n}-gram'] = p
            if p > 0:
                log_prec_sum += math.log(p)
        else:
            precisions[f'{n}-gram'] = 0.0
    
    geo_mean = math.exp(log_prec_sum / effective_n) if effective_n > 0 else 0.0
    bp = brevity_penalty(cand_tokens, ref_tokens)
    
    return {
        'bleu': bp * geo_mean,
        'bp': bp,
        **precisions
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("BLEU Score Calculator")
    print("=" * 50)
    
    print("\nChoose display mode:")
    print("  [1] Normal BLEU score only")
    print("  [2] Detailed n-gram breakdown (1-gram, 2-gram, 3-gram, 4-gram)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    show_ngrams = (choice == "2")
    
    examples = [
        {
            "candidate": "the cat sat on the mat",
            "references": ["the cat sat on the mat"],
            "label": "Perfect match"
        },
        {
            "candidate": "the cat on the mat",
            "references": ["the cat sat on the mat"],
            "label": "Missing 'sat'"
        },
        {
            "candidate": "a cat is sitting on a mat",
            "references": ["the cat sat on the mat", "there is a cat on the mat"],
            "label": "Paraphrase (multi-ref)"
        },
    ]
    
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    
    for ex in examples:
        print(f"\n{ex['label']}")
        print(f"  Candidate:  {ex['candidate']}")
        print(f"  Reference:  {ex['references']}")
        
        if show_ngrams:
            details = bleu_ngram_details(ex["candidate"], ex["references"])
            print(f"  1-gram:     {details['1-gram']:.4f}")
            print(f"  2-gram:     {details['2-gram']:.4f}")
            print(f"  3-gram:     {details['3-gram']:.4f}")
            print(f"  4-gram:     {details['4-gram']:.4f}")
            print(f"  BP:         {details['bp']:.4f}")
            print(f"  BLEU:       {details['bleu']:.4f}")
        else:
            score = bleu(ex["candidate"], ex["references"])
            print(f"  BLEU:       {score:.4f}")
    
    print("\n" + "=" * 50)
