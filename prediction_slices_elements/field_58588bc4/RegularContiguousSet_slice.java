// Source-based slice around line 40
// Method: com.google.common.collect.RegularContiguousSet.range


/**
 * An implementation of {@link ContiguousSet} that contains one or more elements.
 *
 * @author Gregory Kick
 */
@GwtCompatible
@SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
final class RegularContiguousSet<C extends Comparable> extends ContiguousSet<C> {
  private final Range<C> range;

  RegularContiguousSet(Range<C> range, DiscreteDomain<C> domain) {
    super(domain);
    this.range = range;
  }

  private ContiguousSet<C> intersectionInCurrentDomain(Range<C> other) {
    return range.isConnected(other)
        ? ContiguousSet.create(range.intersection(other), domain)
        : new EmptyContiguousSet<C>(domain);
