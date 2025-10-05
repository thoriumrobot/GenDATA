// Source-based slice around line 58
// Method: <com.google.common.graph.ImmutableNetwork: ImmutableNetwork copyOf(Network)>

@SuppressWarnings("Immutable") // Extends StandardNetwork but uses ImmutableMaps.
public final class ImmutableNetwork<N, E> extends StandardNetwork<N, E> {

  private ImmutableNetwork(Network<N, E> network) {
    super(
        NetworkBuilder.from(network), getNodeConnections(network), getEdgeToReferenceNode(network));
  }

  /** Returns an immutable copy of {@code network}. */
  public static <N, E> ImmutableNetwork<N, E> copyOf(Network<N, E> network) {
    return (network instanceof ImmutableNetwork)
        ? (ImmutableNetwork<N, E>) network
        : new ImmutableNetwork<N, E>(network);
  }

  /**
   * Simply returns its argument.
   *
   * @deprecated no need to use this
   */
