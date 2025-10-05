// Source-based slice around line 52
// Method: com.google.common.collect.EmptyImmutableSetMultimap.serialVersionUID

  @Override
  public ImmutableMap<Object, Collection<Object>> asMap() {
    return super.asMap();
  }

  private Object readResolve() {
    return INSTANCE; // preserve singleton property
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
