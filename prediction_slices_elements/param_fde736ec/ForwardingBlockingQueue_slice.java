// Source-based slice around line 53
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: int drainTo(Collection,int)>


  /** Constructor for use by subclasses. */
  protected ForwardingBlockingQueue() {}

  @Override
  protected abstract BlockingQueue<E> delegate();

  @CanIgnoreReturnValue
  @Override
  public int drainTo(Collection<? super E> c, int maxElements) {
    return delegate().drainTo(c, maxElements);
  }

  @CanIgnoreReturnValue
  @Override
  public int drainTo(Collection<? super E> c) {
    return delegate().drainTo(c);
  }

  @CanIgnoreReturnValue // TODO(kak): consider removing this
