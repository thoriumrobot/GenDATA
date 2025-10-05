// Source-based slice around line 27
// Method: com.google.common.graph.AbstractGraphBuilder.directed


import com.google.common.base.Optional;

/**
 * A base class for builders that construct graphs with user-defined properties.
 *
 * @author James Sexton
 */
abstract class AbstractGraphBuilder<N> {
  final boolean directed;
  boolean allowsSelfLoops = false;
  ElementOrder<N> nodeOrder = ElementOrder.insertion();
  ElementOrder<N> incidentEdgeOrder = ElementOrder.unordered();

  Optional<Integer> expectedNodeCount = Optional.absent();

  /**
   * Creates a new instance with the specified edge directionality.
   *
   * @param directed if true, creates an instance for graphs whose edges are each directed; if
