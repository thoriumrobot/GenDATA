// Source-based slice around line 2309
// Method: <com.google.common.collect.MapMakerInternalMap: Equivalence valueEquivalence()>

    return entryHelper.keyStrength();
  }

  @VisibleForTesting
  Strength valueStrength() {
    return entryHelper.valueStrength();
  }

  @VisibleForTesting
  Equivalence<Object> valueEquivalence() {
    return entryHelper.valueStrength().defaultEquivalence();
  }

  // ConcurrentMap methods

  @Override
  public boolean isEmpty() {
    /*
     * Sum per-segment modCounts to avoid mis-reporting when elements are concurrently added and
     * removed in one segment while checking another, in which case the table was never actually
