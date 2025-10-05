// Source-based slice around line 311
// Method: <com.google.common.graph.Network: int degree(N)>

   * <p>For directed networks, this is equal to {@code inDegree(node) + outDegree(node)}.
   *
   * <p>For undirected networks, this is equal to {@code incidentEdges(node).size()} + (number of
   * self-loops incident to {@code node}).
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  int degree(N node);

  /**
   * Returns the count of {@code node}'s {@link #inEdges(Object) incoming edges} in a directed
   * network. In an undirected network, returns the {@link #degree(Object)}.
   *
   * <p>If the count is greater than {@code Integer.MAX_VALUE}, returns {@code Integer.MAX_VALUE}.
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this network
   */
  int inDegree(N node);
