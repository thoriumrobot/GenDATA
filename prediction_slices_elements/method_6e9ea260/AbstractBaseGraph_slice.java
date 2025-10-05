// Source-based slice around line 52
// Method: <com.google.common.graph.AbstractBaseGraph: long edgeCount()>

 * @param <N> Node parameter type
 */
abstract class AbstractBaseGraph<N> implements BaseGraph<N> {

  /**
   * Returns the number of edges in this graph; used to calculate the size of {@link Graph#edges()}.
   * This implementation requires O(|N|) time. Classes extending this one may manually keep track of
   * the number of edges as the graph is updated, and override this method for better performance.
   */
  protected long edgeCount() {
    long degreeSum = 0L;
    for (N node : nodes()) {
      degreeSum += degree(node);
    }
    // According to the degree sum formula, this is equal to twice the number of edges.
    checkState((degreeSum & 1) == 0);
    return degreeSum >>> 1;
  }

  /**
