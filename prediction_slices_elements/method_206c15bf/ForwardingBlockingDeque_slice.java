// Source-based slice around line 60
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: int remainingCapacity()>

    implements BlockingDeque<E> {

  /** Constructor for use by subclasses. */
  protected ForwardingBlockingDeque() {}

  @Override
  protected abstract BlockingDeque<E> delegate();

  @Override
  public int remainingCapacity() {
    return delegate().remainingCapacity();
  }

  @Override
  public void putFirst(E e) throws InterruptedException {
    delegate().putFirst(e);
  }

  @Override
  public void putLast(E e) throws InterruptedException {
