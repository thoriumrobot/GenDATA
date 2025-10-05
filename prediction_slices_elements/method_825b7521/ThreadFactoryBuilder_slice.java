// Source-based slice around line 89
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: ThreadFactoryBuilder setNameFormat(String)>

   *
   * @param nameFormat a {@link String#format(String, Object...)}-compatible format String, to which
   *     a unique integer (0, 1, etc.) will be supplied as the single parameter. This integer will
   *     be unique to the built instance of the ThreadFactory and will be assigned sequentially. For
   *     example, {@code "rpc-pool-%d"} will generate thread names like {@code "rpc-pool-0"}, {@code
   *     "rpc-pool-1"}, {@code "rpc-pool-2"}, etc.
   * @return this for the builder pattern
   */
  @CanIgnoreReturnValue
  public ThreadFactoryBuilder setNameFormat(String nameFormat) {
    String unused = format(nameFormat, 0); // fail fast if the format is bad or null
    this.nameFormat = nameFormat;
    return this;
  }

  /**
   * Sets daemon or not for new threads created with this ThreadFactory.
   *
   * <p><b>Java 21+ users:</b> use {@link Thread.Builder.OfPlatform#daemon(boolean)} instead.
   *
