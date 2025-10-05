// Source-based slice around line 304
// Method: <com.google.common.util.concurrent.AbstractService: void awaitRunning()>

      } finally {
        monitor.leave();
        dispatchListenerEvents();
      }
    }
    return this;
  }

  @Override
  public final void awaitRunning() {
    monitor.enterWhenUninterruptibly(hasReachedRunning);
    try {
      checkCurrentState(RUNNING);
    } finally {
      monitor.leave();
    }
  }

  /**
   * @since 28.0
