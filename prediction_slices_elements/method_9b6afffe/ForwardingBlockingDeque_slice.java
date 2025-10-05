// Source-based slice around line 57
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: BlockingDeque delegate()>

@J2ktIncompatible
@GwtIncompatible
public abstract class ForwardingBlockingDeque<E> extends ForwardingDeque<E>
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
