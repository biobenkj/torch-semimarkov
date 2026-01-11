# Parameter guide: T, K, C

This guide expands on the three core semi-CRF parameters and how they map to
common genomics setups.

## T = Sequence length

T is the sequence length in positions. In genomics this could be:
- base pairs (1 bp = 1 position)
- tokens (for example 4/8/16 bp per token, k-mers, pooled stride)

Intuition: T is the width of context you decode in one shot.

Examples:
- Single-gene locus decoding: T might be the span of a gene plus flanks.
- Chunked genome scanning: T is your chunk size (with overlaps).
- Transcript-level decoding: T might be the span covering one transcript model.

Why it matters:
- Larger T uses more context and reduces edge effects.
- Vectorized linear scan time grows roughly linearly with T.

## K = Maximum segment duration

Semi-CRFs predict segments, not per-base labels. K is the max duration you
consider when forming a segment that ends at position t.

In genomics, segment lengths correspond to:
- Exon length
- Intron length
- UTR length
- Intergenic/background length
- TE element length (or chunks)

Intuition: K sets the longest one-piece region the decoder can model without
splitting it.

Common strategies:
- Cap and split background-like labels (long regions become multiple segments).
- Use label-specific K values (shorter for exons/UTRs, longer for introns).

In practice, pick K using a quantile (p95/p99) of observed lengths rather than
max length.

## C = Number of segment labels

C is the label set size, your segment-level annotation vocabulary.

Common choices:
- Coarse (C ~ 3): exon, intron, intergenic/background
- Gene-structure (C ~ 4-8): split exon into CDS/UTR, plus intron, intergenic
- Rich (C ~ 10-30+): strand-split labels, more biotypes, TE classes, signals

Intuition: C controls how detailed the segmentation is.

## The decoder's decision at each step

At the end of position t, the semi-CRF asks:
- Did a segment end here?
- If yes, which label c (one of C) and which duration d (1..K)?
- What label did we transition from?

So:
- T controls how many decisions you make.
- K controls how far back you can look.
- C controls how many segment types exist.

## Practical examples

Gene structure annotation:
- T = locus/chunk length you decode in one shot
- K = max exon/intron/background segment without splitting
- C = exon/intron/UTR/etc label set

TE annotation:
- T = chunk length
- K = max TE segment length you treat as one element (or chunk)
- C = TE families/superfamilies plus background

## Choosing parameters

1. Pick T based on your inference unit (gene locus or genome chunk).
2. Pick C based on your desired label granularity.
3. Pick K based on segment lengths you want to model as single segments.
