// Source-based slice around line 2304
// Method: <com.google.common.collect.MapMakerInternalMap: Strength valueStrength()>

    }
  }

  @VisibleForTesting
  Strength keyStrength() {
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

