// Source-based slice around line 132
// Method: <com.google.common.collect.Range: Range create(Cut,Cut)>

@GwtCompatible
@SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
@Immutable(containerOf = "C")
public final class Range<C extends Comparable> implements Predicate<C>, Serializable {
  @SuppressWarnings("unchecked")
  static <C extends Comparable<?>> Ordering<Range<C>> rangeLexOrdering() {
    return (Ordering<Range<C>>) RangeLexOrdering.INSTANCE;
  }

  static <C extends Comparable<?>> Range<C> create(Cut<C> lowerBound, Cut<C> upperBound) {
    return new Range<>(lowerBound, upperBound);
  }

  /**
   * Returns a range that contains all values strictly greater than {@code lower} and strictly less
   * than {@code upper}.
   *
   * @throws IllegalArgumentException if {@code lower} is greater than <i>or equal to</i> {@code
   *     upper}
   * @throws ClassCastException if {@code lower} and {@code upper} are not mutually comparable
