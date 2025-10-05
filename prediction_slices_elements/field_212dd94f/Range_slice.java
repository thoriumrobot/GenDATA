// Source-based slice around line 324
// Method: com.google.common.collect.Range.lowerBound

    C max = min;
    while (valueIterator.hasNext()) {
      C value = checkNotNull(valueIterator.next());
      min = Ordering.<C>natural().min(min, value);
      max = Ordering.<C>natural().max(max, value);
    }
    return closed(min, max);
  }

  final Cut<C> lowerBound;
  final Cut<C> upperBound;

  private Range(Cut<C> lowerBound, Cut<C> upperBound) {
    this.lowerBound = checkNotNull(lowerBound);
    this.upperBound = checkNotNull(upperBound);
    if (lowerBound.compareTo(upperBound) > 0
        || lowerBound == Cut.<C>aboveAll()
        || upperBound == Cut.<C>belowAll()) {
      throw new IllegalArgumentException("Invalid range: " + toString(lowerBound, upperBound));
    }
