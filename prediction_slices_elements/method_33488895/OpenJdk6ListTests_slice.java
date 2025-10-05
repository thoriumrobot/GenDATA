// Source-based slice around line 49
// Method: <com.google.common.collect.testing.OpenJdk6ListTests: Collection suppressForCheckedList()>

    return new OpenJdk6ListTests().allTests();
  }

  @Override
  protected Collection<Method> suppressForArraysAsList() {
    return asList(getToArrayIsPlainObjectArrayMethod());
  }

  @Override
  protected Collection<Method> suppressForCheckedList() {
    return asList(
        CollectionAddTester.getAddNullSupportedMethod(),
        getAddSupportedNullPresentMethod(),
        ListAddAtIndexTester.getAddNullSupportedMethod(),
        getSetNullSupportedMethod());
  }
}
