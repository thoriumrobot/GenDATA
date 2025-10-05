// Source-based slice around line 202
// Method: <com.google.common.primitives.ImmutableIntArray: Builder builder()>


  /**
   * Returns a new, empty builder for {@link ImmutableIntArray} instances, with a default initial
   * capacity. The returned builder is not thread-safe.
   *
   * <p><b>Performance note:</b> The {@link ImmutableIntArray} that is built will very likely occupy
   * more memory than necessary; to trim memory usage, build using {@code
   * builder.build().trimmed()}.
   */
  public static Builder builder() {
    return new Builder(10);
  }

  /**
   * A builder for {@link ImmutableIntArray} instances; obtained using {@link
   * ImmutableIntArray#builder}.
   */
  public static final class Builder {
    private int[] array;
    private int count = 0; // <= array.length
