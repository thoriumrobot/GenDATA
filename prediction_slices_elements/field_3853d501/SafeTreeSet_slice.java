// Source-based slice around line 47
// Method: com.google.common.collect.testing.SafeTreeSet.delegate

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
    this(new TreeSet<E>());
  }

  public SafeTreeSet(Collection<? extends E> collection) {
    this(new TreeSet<E>(collection));
  }

  public SafeTreeSet(Comparator<? super E> comparator) {
