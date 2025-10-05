// Source-based slice around line 39
// Method: com.google.common.collect.testing.SafeTreeSet.NATURAL_ORDER

/**
 * A wrapper around {@code TreeSet} that aggressively checks to see if elements are mutually
 * comparable. This implementation passes the navigable set test suites.
 *
 * @author Louis Wasserman
 */
@GwtIncompatible
public final class SafeTreeSet<E> implements Serializable, NavigableSet<E> {
  @SuppressWarnings("unchecked")
  private static final Comparator<Object> NATURAL_ORDER =
      new Comparator<Object>() {
        @Override
        public int compare(Object o1, Object o2) {
          return ((Comparable<Object>) o1).compareTo(o2);
        }
      };

  private final NavigableSet<E> delegate;

  public SafeTreeSet() {
