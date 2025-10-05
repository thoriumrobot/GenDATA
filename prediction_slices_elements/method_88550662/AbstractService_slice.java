// Source-based slice around line 339
// Method: <com.google.common.util.concurrent.AbstractService: void awaitTerminated()>

      // It is possible due to races that we are currently in the expected state even though we
      // timed out. e.g. if we weren't event able to grab the lock within the timeout we would never
      // even check the guard. I don't think we care too much about this use case but it could lead
      // to a confusing error message.
      throw new TimeoutException("Timed out waiting for " + this + " to reach the RUNNING state.");
    }
  }

  @Override
  public final void awaitTerminated() {
    monitor.enterWhenUninterruptibly(isStopped);
    try {
      checkCurrentState(TERMINATED);
    } finally {
      monitor.leave();
    }
  }

  /**
   * @since 28.0
