// Source-based slice around line 112
// Method: <com.google.common.graph.MutableNetwork: boolean removeEdge(E)>

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
