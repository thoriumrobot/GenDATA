// Source-based slice around line 39
// Method: com.google.common.collect.RegularImmutableSortedMultiset.NATURAL_EMPTY_MULTISET

 * An immutable sorted multiset with one or more distinct elements.
 *
 * @author Louis Wasserman
 */
@SuppressWarnings("serial") // uses writeReplace, not default serialization
@GwtIncompatible
final class RegularImmutableSortedMultiset<E> extends ImmutableSortedMultiset<E> {
  private static final long[] zeroCumulativeCounts = {0};

  static final ImmutableSortedMultiset<?> NATURAL_EMPTY_MULTISET =
      new RegularImmutableSortedMultiset<>(Ordering.natural());

  @VisibleForTesting final transient RegularImmutableSortedSet<E> elementSet;
  private final transient long[] cumulativeCounts;
  private final transient int offset;
  private final transient int length;

  RegularImmutableSortedMultiset(Comparator<? super E> comparator) {
    this.elementSet = ImmutableSortedSet.emptySet(comparator);
    this.cumulativeCounts = zeroCumulativeCounts;
