// Source-based slice around line 489
// Method: <com.google.common.cache.CacheBuilderSpec: String format(String,Object)>

  private static final class RefreshDurationParser extends DurationParser {
    @Override
    protected void parseDuration(CacheBuilderSpec spec, long duration, TimeUnit unit) {
      checkArgument(spec.refreshTimeUnit == null, "refreshAfterWrite already set");
      spec.refreshDuration = duration;
      spec.refreshTimeUnit = unit;
    }
  }

  private static String format(String format, Object... args) {
    return String.format(Locale.ROOT, format, args);
  }
}
