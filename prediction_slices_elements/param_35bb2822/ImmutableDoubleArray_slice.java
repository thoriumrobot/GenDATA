// Source-based slice around line 192
// Method: <com.google.common.primitives.ImmutableDoubleArray: Builder builder(int)>

   * Returns a new, empty builder for {@link ImmutableDoubleArray} instances, sized to hold up to
   * {@code initialCapacity} values without resizing. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> When feasible, {@code initialCapacity} should be the exact number
   * of values that will be added, if that knowledge is readily available. It is better to guess a
   * value slightly too high than slightly too low. If the value is not exact, the {@link
   * ImmutableDoubleArray} that is built will very likely occupy more memory than strictly
   * necessary; to trim memory usage, build using {@code builder.build().trimmed()}.
   */
  public static Builder builder(int initialCapacity) {
    checkArgument(initialCapacity >= 0, "Invalid initialCapacity: %s", initialCapacity);
    return new Builder(initialCapacity);
  }

  /**
   * Returns a new, empty builder for {@link ImmutableDoubleArray} instances, with a default initial
   * capacity. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> The {@link ImmutableDoubleArray} that is built will very likely
   * occupy more memory than necessary; to trim memory usage, build using {@code
