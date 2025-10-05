// Source-based slice around line 48
// Method: <com.google.common.collect.EmptyImmutableListMultimap: Object readResolve()>

   * of common.collect a second time with the results of the first compilation on the classpath. Or
   * just back this out once we stop doing that (which we'll do after our internal GWT setup
   * changes).
   */
  @Override
  public ImmutableMap<Object, Collection<Object>> asMap() {
    return super.asMap();
  }

  private Object readResolve() {
    return INSTANCE; // preserve singleton property
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
