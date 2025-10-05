// Source-based slice around line 490
// Method: <com.google.common.util.concurrent.AbstractService: boolean isRunning()>

          break;
      }
    } finally {
      monitor.leave();
      dispatchListenerEvents();
    }
  }

  @Override
  public final boolean isRunning() {
    return state() == RUNNING;
  }

  @Override
  public final State state() {
    return snapshot.externalState();
  }

  /**
   * @since 14.0
