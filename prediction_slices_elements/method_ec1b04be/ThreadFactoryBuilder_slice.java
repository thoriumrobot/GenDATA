// Source-based slice around line 149
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactoryBuilder setUncaughtExceptionHandler(UncaughtExceptionHandler)>

   *
   * <p><b>Java 21+ users:</b> use {@link
   * Thread.Builder#uncaughtExceptionHandler(Thread.UncaughtExceptionHandler)} instead.
   *
   * @param uncaughtExceptionHandler the uncaught exception handler for new Threads created with
   *     this ThreadFactory
   * @return this for the builder pattern
   */
  @CanIgnoreReturnValue
  public ThreadFactoryBuilder setUncaughtExceptionHandler(
      UncaughtExceptionHandler uncaughtExceptionHandler) {
    this.uncaughtExceptionHandler = checkNotNull(uncaughtExceptionHandler);
    return this;
  }

  /**
   * Sets the backing {@link ThreadFactory} for new threads created with this ThreadFactory. Threads
   * will be created by invoking #newThread(Runnable) on this backing {@link ThreadFactory}.
   *
   * @param backingThreadFactory the backing {@link ThreadFactory} which will be delegated to during
