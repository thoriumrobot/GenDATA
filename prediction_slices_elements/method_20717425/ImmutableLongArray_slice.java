// Source-based slice around line 204
// Method: <com.google.common.primitives.ImmutableLongArray: Builder builder()>


  /**
   * Returns a new, empty builder for {@link ImmutableLongArray} instances, with a default initial
   * capacity. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> The {@link ImmutableLongArray} that is built will very likely
   * occupy more memory than necessary; to trim memory usage, build using {@code
   * builder.build().trimmed()}.
   */
  public static Builder builder() {
    return new Builder(10);
  }

  /**
   * A builder for {@link ImmutableLongArray} instances; obtained using {@link
   * ImmutableLongArray#builder}.
   */
  public static final class Builder {
    private long[] array;
    private int count = 0; // <= array.length
