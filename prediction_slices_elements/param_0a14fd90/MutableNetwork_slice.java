// Source-based slice around line 104
// Method: <com.google.common.graph.MutableNetwork: boolean removeNode(N)>

  @CanIgnoreReturnValue
  boolean addEdge(EndpointPair<N> endpoints, E edge);

  /**
   * Removes {@code node} if it is present; all edges incident to {@code node} will also be removed.
   *
   * @return {@code true} if the network was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean removeNode(N node);

  /**
   * Removes {@code edge} from this network, if it is present.
   *
   * @return {@code true} if the network was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean removeEdge(E edge);
}
