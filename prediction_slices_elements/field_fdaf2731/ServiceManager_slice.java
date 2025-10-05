// Source-based slice around line 196
// Method: com.google.common.util.concurrent.ServiceManager.state

  }

  /**
   * An encapsulation of all of the state that is accessed by the {@linkplain ServiceListener
   * service listeners}. This is extracted into its own object so that {@link ServiceListener} could
   * be made {@code static} and its instances can be safely constructed and added in the {@link
   * ServiceManager} constructor without having to close over the partially constructed {@link
   * ServiceManager} instance (i.e. avoid leaking a pointer to {@code this}).
   */
  private final ServiceManagerState state;

  private final ImmutableList<Service> services;

  /**
   * Constructs a new instance for managing the given services.
   *
   * @param services The services to manage
   * @throws IllegalArgumentException if not all services are {@linkplain State#NEW new} or if there
   *     are any duplicate services.
   */
