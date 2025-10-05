// Source-based slice around line 67
// Method: <com.google.common.collect.testing.TestIntegerSetGenerator: List order(List)>

   * <p>By default, returns the supplied elements in their given order; however, generators for
   * containers with a known order other than insertion order must override this method.
   *
   * <p>Note: This default implementation is overkill (but valid) for an unordered container. An
   * equally valid implementation for an unordered container is to throw an exception. The chosen
   * implementation, however, has the advantage of working for insertion-ordered containers, as
   * well.
   */
  @Override
  public List<Integer> order(List<Integer> insertionOrder) {
    return insertionOrder;
  }
}
