// Source-based slice around line 47
// Method: com.google.common.base.Ticker.SYSTEM_TICKER

  /**
   * A ticker that reads the current time using {@link System#nanoTime}.
   *
   * @since 10.0
   */
  public static Ticker systemTicker() {
    return SYSTEM_TICKER;
  }

  private static final Ticker SYSTEM_TICKER =
      new Ticker() {
        @Override
        @SuppressWarnings("GoodTime") // reading system time without TimeSource
        public long read() {
          return System.nanoTime();
        }
      };
}
