// Source-based slice around line 63
// Method: com.google.common.collect.ImmutableRangeSet.EMPTY

 *
 * @author Louis Wasserman
 * @since 14.0
 */
@SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
@GwtIncompatible
public final class ImmutableRangeSet<C extends Comparable> extends AbstractRangeSet<C>
    implements Serializable {

  private static final ImmutableRangeSet<Comparable<?>> EMPTY =
      new ImmutableRangeSet<>(ImmutableList.of());

  private static final ImmutableRangeSet<Comparable<?>> ALL =
      new ImmutableRangeSet<>(ImmutableList.of(Range.all()));

  /**
   * Returns a {@code Collector} that accumulates the input elements into a new {@code
   * ImmutableRangeSet}. As in {@link Builder}, overlapping ranges are not permitted and adjacent
   * ranges will be merged.
   *
