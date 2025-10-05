// Source-based slice around line 535
// Method: <com.google.common.graph.Network: int hashCode()>


  /**
   * Returns the hash code for this network. The hash code of a network is defined as the hash code
   * of a map from each of its {@link #edges() edges} to their {@link #incidentNodes(Object)
   * incident nodes}.
   *
   * <p>A reference implementation of this is provided by {@link AbstractNetwork#hashCode()}.
   */
  @Override
  int hashCode();
}
