// Source-based slice around line 2299
// Method: <com.google.common.collect.MapMakerInternalMap: Strength keyStrength()>

      }

      for (Segment<?, ?, ?, ?> segment : map.segments) {
        segment.runCleanup();
      }
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
