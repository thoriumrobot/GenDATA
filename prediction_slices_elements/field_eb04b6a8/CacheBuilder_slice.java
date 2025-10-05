// Source-based slice around line 230
// Method: com.google.common.cache.CacheBuilder.EMPTY_STATS


            @Override
            public void recordEviction() {}

            @Override
            public CacheStats snapshot() {
              return EMPTY_STATS;
            }
          });
  static final CacheStats EMPTY_STATS = new CacheStats(0, 0, 0, 0, 0, 0);

  /*
   * We avoid using a method reference or lambda here for now:
   *
   * - method reference: Inside Google, CacheBuilder is used from the implementation of a custom
   *   ClassLoader that is sometimes used as a system classloader. That's a problem because
   *   method-reference linking tries to look up the system classloader, and it fails because there
   *   isn't one yet.
   *
   * - lambda: Outside Google, we got a report of a similar problem in
