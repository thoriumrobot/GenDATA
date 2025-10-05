// Source-based slice around line 28
// Method: com.google.common.graph.IncidentEdgeSet.node

import java.util.AbstractSet;
import java.util.Set;
import org.jspecify.annotations.Nullable;

/**
 * Abstract base class for an incident edges set that allows different implementations of {@link
 * AbstractSet#iterator()}.
 */
abstract class IncidentEdgeSet<N> extends AbstractSet<EndpointPair<N>> {
  final N node;
  final BaseGraph<N> graph;

  IncidentEdgeSet(BaseGraph<N> graph, N node) {
    this.graph = graph;
    this.node = node;
  }

  @Override
  public boolean remove(@Nullable Object o) {
    throw new UnsupportedOperationException();
