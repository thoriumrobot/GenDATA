// Source-based slice around line 100
// Method: com.google.common.collect.MapMaker.initialCapacity

public final class MapMaker {
  private static final int DEFAULT_INITIAL_CAPACITY = 16;
  private static final int DEFAULT_CONCURRENCY_LEVEL = 4;

  static final int UNSET_INT = -1;

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
