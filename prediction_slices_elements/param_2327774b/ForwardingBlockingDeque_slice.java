// Source-based slice around line 130
// Method: <com.google.common.util.concurrent.ForwardingBlockingDeque: int drainTo(Collection,int)>

    return delegate().poll(timeout, unit);
  }

  @Override
  public int drainTo(Collection<? super E> c) {
    return delegate().drainTo(c);
  }

  @Override
  public int drainTo(Collection<? super E> c, int maxElements) {
    return delegate().drainTo(c, maxElements);
  }
}
