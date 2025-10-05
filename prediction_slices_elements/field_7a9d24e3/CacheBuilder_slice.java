// Source-based slice around line 268
// Method: com.google.common.cache.CacheBuilder.NULL_TICKER

  enum OneWeigher implements Weigher<Object, Object> {
    INSTANCE;

    @Override
    public int weigh(Object key, Object value) {
      return 1;
    }
  }

  static final Ticker NULL_TICKER =
      new Ticker() {
        @Override
        public long read() {
          return 0;
        }
      };

  // We use a holder class to delay initialization: https://github.com/google/guava/issues/6566
  private static final class LoggerHolder {
    static final Logger logger = Logger.getLogger(CacheBuilder.class.getName());
