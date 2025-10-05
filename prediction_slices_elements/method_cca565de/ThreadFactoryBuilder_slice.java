// Source-based slice around line 179
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactory build()>

  /**
   * Returns a new thread factory using the options supplied during the building process. After
   * building, it is still possible to change the options used to build the ThreadFactory and/or
   * build again. State is not shared amongst built instances.
   *
   * <p><b>Java 21+ users:</b> use {@link Thread.Builder#factory()} instead.
   *
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
