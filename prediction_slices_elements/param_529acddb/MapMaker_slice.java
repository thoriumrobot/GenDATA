// Source-based slice around line 123
// Method: <com.google.common.collect.MapMaker: MapMaker keyEquivalence(Equivalence)>

  /**
   * Sets a custom {@code Equivalence} strategy for comparing keys.
   *
   * <p>By default, the map uses {@link Equivalence#identity} to determine key equality when {@link
   * #weakKeys} is specified, and {@link Equivalence#equals()} otherwise. The only place this is
   * used is in {@link Interners.WeakInterner}.
   */
  @CanIgnoreReturnValue
  @GwtIncompatible // To be supported
  MapMaker keyEquivalence(Equivalence<Object> equivalence) {
    checkState(keyEquivalence == null, "key equivalence was already set to %s", keyEquivalence);
    keyEquivalence = checkNotNull(equivalence);
    this.useCustomMap = true;
    return this;
  }

  Equivalence<Object> getKeyEquivalence() {
    return MoreObjects.firstNonNull(keyEquivalence, getKeyStrength().defaultEquivalence());
  }

