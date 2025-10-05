// Source-based slice around line 81
// Method: <com.google.common.collect.testing.OpenJdk6MapTests: Collection suppressForConcurrentSkipListMap()>

     * feature for entrySet() that supports add(), or something else
     */
    return asList(
        getAddUnsupportedNotPresentMethod(),
        getAddAllUnsupportedNonePresentMethod(),
        getAddAllUnsupportedSomePresentMethod());
  }

  @Override
  protected Collection<Method> suppressForConcurrentSkipListMap() {
    List<Method> methods = new ArrayList<>();
    methods.addAll(super.suppressForConcurrentSkipListMap());
    methods.add(getContainsEntryWithIncomparableKeyMethod());
    methods.add(getContainsEntryWithIncomparableValueMethod());
    return methods;
  }
}
