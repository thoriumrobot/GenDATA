// Source-based slice around line 137
// Method: <com.google.common.graph.AbstractBaseGraph: int degree(N)>

                      graph.adjacentNodes(node).iterator(),
                      (N adjacentNode) -> EndpointPair.unordered(node, adjacentNode)));
            }
          }
        };
    return nodeInvalidatableSet(incident, node);
  }

  @Override
  public int degree(N node) {
    if (isDirected()) {
      return IntMath.saturatedAdd(predecessors(node).size(), successors(node).size());
    } else {
      Set<N> neighbors = adjacentNodes(node);
      int selfLoopCount = (allowsSelfLoops() && neighbors.contains(node)) ? 1 : 0;
      return IntMath.saturatedAdd(neighbors.size(), selfLoopCount);
    }
  }

  @Override
