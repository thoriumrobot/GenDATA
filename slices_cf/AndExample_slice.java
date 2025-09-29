
    @Positive
  private static final @IndexFor("iYearInfoCache") int CACHE_MASK = CACHE_SIZE - 1;

  private static final String[] iYearInfoCache = new String[CACHE_SIZE];

  private String getYearInfo(int year) {
    return iYearInfoCache[year & CACHE_MASK];
  }
}
