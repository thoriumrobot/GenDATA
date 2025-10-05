// Source-based slice around line 26
// Method: com.google.common.graph.GraphConstants.DEFAULT_NODE_COUNT

package com.google.common.graph;

/** A utility class to hold various constants used by the Guava Graph library. */
final class GraphConstants {

  private GraphConstants() {}

  static final int EXPECTED_DEGREE = 2;

  static final int DEFAULT_NODE_COUNT = 10;
  static final int DEFAULT_EDGE_COUNT = DEFAULT_NODE_COUNT * EXPECTED_DEGREE;

  // Load factor and capacity for "inner" (i.e. per node/edge element) hash sets or maps
  static final float INNER_LOAD_FACTOR = 1.0f;
  static final int INNER_CAPACITY = 2; // ceiling(EXPECTED_DEGREE / INNER_LOAD_FACTOR)

  // Error messages
  static final String NODE_NOT_IN_GRAPH = "Node %s is not an element of this graph.";
  static final String EDGE_NOT_IN_GRAPH = "Edge %s is not an element of this graph.";
  static final String NODE_REMOVED_FROM_GRAPH =
