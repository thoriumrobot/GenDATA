// Source-based slice around line 48
// Method: <com.google.common.collect.testing.OpenJdk6QueueTests: Collection suppressForPriorityQueue()>


  private static final List<Method> PQ_SUPPRESS = asList(getCreateWithNullUnsupportedMethod());

  @Override
  protected Collection<Method> suppressForPriorityBlockingQueue() {
    return PQ_SUPPRESS;
  }

  @Override
  protected Collection<Method> suppressForPriorityQueue() {
    return PQ_SUPPRESS;
  }
}
