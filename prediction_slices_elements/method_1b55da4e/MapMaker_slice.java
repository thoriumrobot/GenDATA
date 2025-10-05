// Source-based slice around line 221
// Method: <com.google.common.collect.MapMaker: Strength getKeyStrength()>

    checkState(keyStrength == null, "Key strength was already set to %s", keyStrength);
    keyStrength = checkNotNull(strength);
    if (strength != Strength.STRONG) {
      // STRONG could be used during deserialization.
      useCustomMap = true;
    }
    return this;
  }

  Strength getKeyStrength() {
    return MoreObjects.firstNonNull(keyStrength, Strength.STRONG);
  }

  /**
   * Specifies that each value (not key) stored in the map should be wrapped in a {@link
   * WeakReference} (by default, strong references are used).
   *
   * <p>Weak values will be garbage collected once they are weakly reachable. This makes them a poor
   * candidate for caching.
   *
