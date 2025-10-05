// Source-based slice around line 120
// Method: <com.google.common.graph.ElementOrder: ElementOrder stable()>

   *         <li>{@code inEdges(node)}: Edge insertion order
   *         <li>{@code outEdges(node)}: Edge insertion order
   *         <li>{@code adjacentEdges(edge)}: Stable order
   *         <li>{@code edgesConnecting(nodeU, nodeV)}: Edge insertion order
   *       </ul>
   * </ul>
   *
   * @since 29.0
   */
  public static <S> ElementOrder<S> stable() {
    return new ElementOrder<>(Type.STABLE, null);
  }

  /** Returns an instance which specifies that insertion ordering is guaranteed. */
  public static <S> ElementOrder<S> insertion() {
    return new ElementOrder<>(Type.INSERTION, null);
  }

  /**
   * Returns an instance which specifies that the natural ordering of the elements is guaranteed.
