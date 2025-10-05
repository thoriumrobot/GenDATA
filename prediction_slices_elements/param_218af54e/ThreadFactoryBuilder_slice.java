// Source-based slice around line 121
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactoryBuilder setPriority(int)>

   * <p><b>Warning:</b> relying on the thread scheduler is <a
   * href="http://errorprone.info/bugpattern/ThreadPriorityCheck">discouraged</a>.
   *
   * <p><b>Java 21+ users:</b> use {@link Thread.Builder.OfPlatform#priority(int)} instead.
   *
   * @param priority the priority for new Threads created with this ThreadFactory
   * @return this for the builder pattern
   */
  @CanIgnoreReturnValue
  public ThreadFactoryBuilder setPriority(int priority) {
    // Thread#setPriority() already checks for validity. These error messages
    // are nicer though and will fail-fast.
    checkArgument(
        priority >= Thread.MIN_PRIORITY,
        "Thread priority (%s) must be >= %s",
        priority,
        Thread.MIN_PRIORITY);
    checkArgument(
        priority <= Thread.MAX_PRIORITY,
        "Thread priority (%s) must be <= %s",
