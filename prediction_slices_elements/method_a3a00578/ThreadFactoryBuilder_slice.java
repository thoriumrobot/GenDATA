// Source-based slice around line 165
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactoryBuilder setThreadFactory(ThreadFactory)>

   * Sets the backing {@link ThreadFactory} for new threads created with this ThreadFactory. Threads
   * will be created by invoking #newThread(Runnable) on this backing {@link ThreadFactory}.
   *
   * @param backingThreadFactory the backing {@link ThreadFactory} which will be delegated to during
   *     thread creation.
   * @return this for the builder pattern
   * @see MoreExecutors
   */
  @CanIgnoreReturnValue
  public ThreadFactoryBuilder setThreadFactory(ThreadFactory backingThreadFactory) {
    this.backingThreadFactory = checkNotNull(backingThreadFactory);
    return this;
  }

  /**
   * Returns a new thread factory using the options supplied during the building process. After
   * building, it is still possible to change the options used to build the ThreadFactory and/or
   * build again. State is not shared amongst built instances.
   *
   * <p><b>Java 21+ users:</b> use {@link Thread.Builder#factory()} instead.
