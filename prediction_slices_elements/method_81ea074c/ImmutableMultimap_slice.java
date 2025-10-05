// Source-based slice around line 129
// Method: <com.google.common.collect.ImmutableMultimap: Builder builder()>

    return ImmutableListMultimap.of(k1, v1, k2, v2, k3, v3, k4, v4, k5, v5);
  }

  // looking for of() with > 5 entries? Use the builder instead.

  /**
   * Returns a new builder. The generated builder is equivalent to the builder created by the {@link
   * Builder} constructor.
   */
  public static <K, V> Builder<K, V> builder() {
    return new Builder<>();
  }

  /**
   * Returns a new builder with a hint for how many distinct keys are expected to be added. The
   * generated builder is equivalent to that returned by {@link #builder}, but may perform better if
   * {@code expectedKeys} is a good estimate.
   *
   * @throws IllegalArgumentException if {@code expectedKeys} is negative
   * @since 33.3.0
