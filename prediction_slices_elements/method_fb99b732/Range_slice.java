// Source-based slice around line 128
// Method: <com.google.common.collect.Range: Ordering rangeLexOrdering()>

 * @author Kevin Bourrillion
 * @author Gregory Kick
 * @since 10.0
 */
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
