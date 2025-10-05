// Source-based slice around line 52
// Method: <com.google.common.collect.testing.OpenJdk6SetTests: Collection suppressForCheckedSet()>

  @Override
  protected Collection<Method> suppressForTreeSetNatural() {
    return asList(
        getAddNullUnsupportedMethod(),
        getAddAllNullUnsupportedMethod(),
        getCreateWithNullUnsupportedMethod());
  }

  @Override
  protected Collection<Method> suppressForCheckedSet() {
    return asList(getAddNullSupportedMethod(), getAddSupportedNullPresentMethod());
  }
}
