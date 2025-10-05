// Source-based slice around line 351
// Method: com.google.common.primitives.ImmutableDoubleArray.start

  @SuppressWarnings("Immutable")
  private final double[] array;

  /*
   * TODO(kevinb): evaluate the trade-offs of going bimorphic to save these two fields from most
   * instances. Note that the instances that would get smaller are the right set to care about
   * optimizing, because the rest have the option of calling `trimmed`.
   */

  private final transient int start; // it happens that we only serialize instances where this is 0
  private final int end; // exclusive

  private ImmutableDoubleArray(double[] array) {
    this(array, 0, array.length);
  }

  private ImmutableDoubleArray(double[] array, int start, int end) {
    this.array = array;
    this.start = start;
    this.end = end;
