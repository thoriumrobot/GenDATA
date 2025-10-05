// Source-based slice around line 141
// Method: <com.google.common.collect.ImmutableMultimap: Builder builderWithExpectedKeys(int)>


  /**
   * Returns a new builder with a hint for how many distinct keys are expected to be added. The
   * generated builder is equivalent to that returned by {@link #builder}, but may perform better if
   * {@code expectedKeys} is a good estimate.
   *
   * @throws IllegalArgumentException if {@code expectedKeys} is negative
   * @since 33.3.0
   */
  public static <K, V> Builder<K, V> builderWithExpectedKeys(int expectedKeys) {
    checkNonnegative(expectedKeys, "expectedKeys");
    return new Builder<>(expectedKeys);
  }

  /**
   * A builder for creating immutable multimap instances, especially {@code public static final}
   * multimaps ("constant multimaps"). Example:
   *
   * {@snippet :
   * static final Multimap<String, Integer> STRING_TO_INTEGER_MULTIMAP =
