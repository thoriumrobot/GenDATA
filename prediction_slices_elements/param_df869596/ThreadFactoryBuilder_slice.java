// Source-based slice around line 186
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactory doBuild(ThreadFactoryBuilder)>

   * @return the fully constructed {@link ThreadFactory}
   */
  public ThreadFactory build() {
    return doBuild(this);
  }

  // Split out so that the anonymous ThreadFactory can't contain a reference back to the builder.
  // At least, I assume that's why. TODO(cpovirk): Check, and maybe add a test for this.
  @SuppressWarnings("ThreadPriorityCheck") // We only propagate user requests (which we discourage).
  private static ThreadFactory doBuild(ThreadFactoryBuilder builder) {
    String nameFormat = builder.nameFormat;
    Boolean daemon = builder.daemon;
    Integer priority = builder.priority;
    UncaughtExceptionHandler uncaughtExceptionHandler = builder.uncaughtExceptionHandler;
    ThreadFactory backingThreadFactory =
        (builder.backingThreadFactory != null)
            ? builder.backingThreadFactory
            : defaultThreadFactory();
    AtomicLong count = (nameFormat != null) ? new AtomicLong(0) : null;
    return new ThreadFactory() {
