// Source-based slice around line 181
// Method: com.google.common.util.concurrent.AbstractService.listeners

    }

    @Override
    public boolean isSatisfied() {
      return state().compareTo(TERMINATED) >= 0;
    }
  }

  /** The listeners to notify during a state transition. */
  private final ListenerCallQueue<Listener> listeners = new ListenerCallQueue<>();

  /**
   * The current state of the service. This should be written with the lock held but can be read
   * without it because it is an immutable object in a volatile field. This is desirable so that
   * methods like {@link #state}, {@link #failureCause} and notably {@link #toString} can be run
   * without grabbing the lock.
   *
   * <p>To update this field correctly the lock must be held to guarantee that the state is
   * consistent.
   */
