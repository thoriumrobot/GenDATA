// Source-based slice around line 638
// Method: <com.google.common.cache.CacheBuilder: Strength getKeyStrength()>

  }

  @CanIgnoreReturnValue
  CacheBuilder<K, V> setKeyStrength(Strength strength) {
    checkState(keyStrength == null, "Key strength was already set to %s", keyStrength);
    keyStrength = checkNotNull(strength);
    return this;
  }

  Strength getKeyStrength() {
    return MoreObjects.firstNonNull(keyStrength, Strength.STRONG);
  }

  /**
   * Specifies that each value (not key) stored in the cache should be wrapped in a {@link
   * WeakReference} (by default, strong references are used).
   *
   * <p>Weak values will be garbage collected once they are weakly reachable. This makes them a poor
   * candidate for caching; consider {@link #softValues} instead.
   *
