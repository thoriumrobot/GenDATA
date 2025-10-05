// Source-based slice around line 713
// Method: <com.google.common.collect.Range: int compareOrThrow(Comparable,Comparable)>

  Object readResolve() {
    if (this.equals(ALL)) {
      return all();
    } else {
      return this;
    }
  }

  @SuppressWarnings("unchecked") // this method may throw CCE
  static int compareOrThrow(Comparable left, Comparable right) {
    return left.compareTo(right);
  }

  /** Needed to serialize sorted collections of Ranges. */
  private static final class RangeLexOrdering extends Ordering<Range<?>> implements Serializable {
    static final Ordering<?> INSTANCE = new RangeLexOrdering();

    @Override
    public int compare(Range<?> left, Range<?> right) {
      return ComparisonChain.start()
