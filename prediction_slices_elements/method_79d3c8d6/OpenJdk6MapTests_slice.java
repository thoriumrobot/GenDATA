// Source-based slice around line 62
// Method: <com.google.common.collect.testing.OpenJdk6MapTests: Collection suppressForConcurrentHashMap()>

        getPutNullKeyUnsupportedMethod(),
        getPutAllNullKeyUnsupportedMethod(),
        getCreateWithNullKeyUnsupportedMethod(),
        getCreateWithNullUnsupportedMethod(), // for keySet
        getContainsEntryWithIncomparableKeyMethod(),
        getContainsEntryWithIncomparableValueMethod());
  }

  @Override
  protected Collection<Method> suppressForConcurrentHashMap() {
    /*
     * The entrySet() of ConcurrentHashMap, unlike that of other Map
     * implementations, supports add() under JDK8. This seems problematic, but I
     * didn't see that discussed in the review, which included many other
     * changes: https://mail.openjdk.org/pipermail/core-libs-dev/2013-May/thread.html#17367
     *
     * TODO(cpovirk): decide what the best long-term action here is: force users
     * to suppress (as we do now), stop testing entrySet().add() at all, make
     * entrySet().add() tests tolerant of either behavior, introduce a map
     * feature for entrySet() that supports add(), or something else
