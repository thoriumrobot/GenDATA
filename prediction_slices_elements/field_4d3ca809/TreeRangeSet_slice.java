// Source-based slice around line 48
// Method: com.google.common.collect.TreeRangeSet.rangesByLowerBound

 * An implementation of {@link RangeSet} backed by a {@link TreeMap}.
 *
 * @author Louis Wasserman
 * @since 14.0
 */
@GwtIncompatible // uses NavigableMap
public class TreeRangeSet<C extends Comparable<?>> extends AbstractRangeSet<C>
    implements Serializable {

  @VisibleForTesting final NavigableMap<Cut<C>, Range<C>> rangesByLowerBound;

  /** Creates an empty {@code TreeRangeSet} instance. */
  public static <C extends Comparable<?>> TreeRangeSet<C> create() {
    return new TreeRangeSet<>(new TreeMap<Cut<C>, Range<C>>());
  }

  /** Returns a {@code TreeRangeSet} initialized with the ranges in the specified range set. */
  public static <C extends Comparable<?>> TreeRangeSet<C> create(RangeSet<C> rangeSet) {
    TreeRangeSet<C> result = create();
    result.addAll(rangeSet);
