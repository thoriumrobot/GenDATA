// Source-based slice around line 268
// Method: <com.google.common.collect.MapMaker: Strength getValueStrength()>

    checkState(valueStrength == null, "Value strength was already set to %s", valueStrength);
    valueStrength = checkNotNull(strength);
    if (strength != Strength.STRONG) {
      // STRONG could be used during deserialization.
      useCustomMap = true;
    }
    return this;
  }

  Strength getValueStrength() {
    return MoreObjects.firstNonNull(valueStrength, Strength.STRONG);
  }

  /**
   * Builds a thread-safe map. This method does not alter the state of this {@code MapMaker}
   * instance, so it can be invoked again to create multiple independent maps.
   *
   * <p>The bulk operations {@code putAll}, {@code equals}, and {@code clear} are not guaranteed to
   * be performed atomically on the returned map. Additionally, {@code size} and {@code
   * containsValue} are implemented as bulk read operations, and thus may fail to observe concurrent
