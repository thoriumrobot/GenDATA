// Source-based slice around line 49
// Method: <com.google.common.util.concurrent.ForwardingBlockingQueue: BlockingQueue delegate()>

@J2ktIncompatible
@GwtIncompatible
public abstract class ForwardingBlockingQueue<E> extends ForwardingQueue<E>
    implements BlockingQueue<E> {

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
