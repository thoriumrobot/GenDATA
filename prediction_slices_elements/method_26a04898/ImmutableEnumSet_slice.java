// Source-based slice around line 68
// Method: <com.google.common.collect.ImmutableEnumSet: boolean isPartialView()>

   * immutability. Hence, we support only {@link EnumSet}.
   */
  private final transient EnumSet<E> delegate;

  private ImmutableEnumSet(EnumSet<E> delegate) {
    this.delegate = delegate;
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
    return Iterators.unmodifiableIterator(delegate.iterator());
  }

  @Override
  public Spliterator<E> spliterator() {
