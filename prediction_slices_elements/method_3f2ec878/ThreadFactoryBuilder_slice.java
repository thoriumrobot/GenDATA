// Source-based slice around line 104
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactoryBuilder setDaemon(boolean)>

  /**
   * Sets daemon or not for new threads created with this ThreadFactory.
   *
   * <p><b>Java 21+ users:</b> use {@link Thread.Builder.OfPlatform#daemon(boolean)} instead.
   *
   * @param daemon whether or not new Threads created with this ThreadFactory will be daemon threads
   * @return this for the builder pattern
   */
  @CanIgnoreReturnValue
  public ThreadFactoryBuilder setDaemon(boolean daemon) {
    this.daemon = daemon;
    return this;
  }

  /**
   * Sets the priority for new threads created with this ThreadFactory.
   *
   * <p><b>Warning:</b> relying on the thread scheduler is <a
   * href="http://errorprone.info/bugpattern/ThreadPriorityCheck">discouraged</a>.
   *
