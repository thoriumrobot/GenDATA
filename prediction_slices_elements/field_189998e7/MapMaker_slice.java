// Source-based slice around line 106
// Method: com.google.common.collect.MapMaker.keyEquivalence

  // TODO(kevinb): dispense with this after benchmarking
  boolean useCustomMap;

  int initialCapacity = UNSET_INT;
  int concurrencyLevel = UNSET_INT;

  @Nullable Strength keyStrength;
  @Nullable Strength valueStrength;

  @Nullable Equivalence<Object> keyEquivalence;

  /**
   * Constructs a new {@code MapMaker} instance with default settings, including strong keys, strong
   * values, and no automatic eviction of any kind.
   */
  public MapMaker() {}

  /**
   * Sets a custom {@code Equivalence} strategy for comparing keys.
   *
