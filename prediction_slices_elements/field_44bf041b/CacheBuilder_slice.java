// Source-based slice around line 244
// Method: com.google.common.cache.CacheBuilder.CACHE_STATS_COUNTER

   * - method reference: Inside Google, CacheBuilder is used from the implementation of a custom
   *   ClassLoader that is sometimes used as a system classloader. That's a problem because
   *   method-reference linking tries to look up the system classloader, and it fails because there
   *   isn't one yet.
   *
   * - lambda: Outside Google, we got a report of a similar problem in
   *   https://github.com/google/guava/issues/6565
   */
  @SuppressWarnings("AnonymousToLambda")
  static final Supplier<StatsCounter> CACHE_STATS_COUNTER =
      new Supplier<StatsCounter>() {
        @Override
        public StatsCounter get() {
          return new SimpleStatsCounter();
        }
      };

  enum NullListener implements RemovalListener<Object, Object> {
    INSTANCE;

