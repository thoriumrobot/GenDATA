// Source-based slice around line 31
// Method: <com.google.common.collect.AbstractRangeSet: boolean contains(C)>

 *
 * @author Louis Wasserman
 */
@SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
@GwtIncompatible
abstract class AbstractRangeSet<C extends Comparable> implements RangeSet<C> {
  AbstractRangeSet() {}

  @Override
  public boolean contains(C value) {
    return rangeContaining(value) != null;
  }

  @Override
  public abstract @Nullable Range<C> rangeContaining(C value);

  @Override
  public boolean isEmpty() {
    return asRanges().isEmpty();
  }
