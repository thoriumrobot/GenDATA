// Source-based slice around line 348
// Method: com.google.common.primitives.ImmutableIntArray.start

  @SuppressWarnings("Immutable")
  private final int[] array;

  /*
   * TODO(kevinb): evaluate the trade-offs of going bimorphic to save these two fields from most
   * instances. Note that the instances that would get smaller are the right set to care about
   * optimizing, because the rest have the option of calling `trimmed`.
   */

  private final transient int start; // it happens that we only serialize instances where this is 0
  private final int end; // exclusive

  private ImmutableIntArray(int[] array) {
    this(array, 0, array.length);
  }

  private ImmutableIntArray(int[] array, int start, int end) {
    this.array = array;
    this.start = start;
    this.end = end;
