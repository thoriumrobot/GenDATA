// Source-based slice around line 205
// Method: <com.google.common.primitives.ImmutableDoubleArray: Builder builder()>


  /**
   * Returns a new, empty builder for {@link ImmutableDoubleArray} instances, with a default initial
   * capacity. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> The {@link ImmutableDoubleArray} that is built will very likely
   * occupy more memory than necessary; to trim memory usage, build using {@code
   * builder.build().trimmed()}.
   */
  public static Builder builder() {
    return new Builder(10);
  }

  /**
   * A builder for {@link ImmutableDoubleArray} instances; obtained using {@link
   * ImmutableDoubleArray#builder}.
   */
  public static final class Builder {
    private double[] array;
    private int count = 0; // <= array.length
