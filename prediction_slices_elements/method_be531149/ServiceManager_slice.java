// Source-based slice around line 353
// Method: <com.google.common.util.concurrent.ServiceManager: void awaitStopped()>

    }
    return this;
  }

  /**
   * Waits for the all the services to reach a terminal state. After this method returns all
   * services will either be {@linkplain Service.State#TERMINATED terminated} or {@linkplain
   * Service.State#FAILED failed}.
   */
  public void awaitStopped() {
    state.awaitStopped();
  }

  /**
   * Waits for the all the services to reach a terminal state for no more than the given time. After
   * this method returns all services will either be {@linkplain Service.State#TERMINATED
   * terminated} or {@linkplain Service.State#FAILED failed}.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if not all of the services have stopped within the deadline
