// Source-based slice around line 66
// Method: <com.google.common.eventbus.Subscriber: void dispatchEvent(Object)>

    this.bus = bus;
    this.target = checkNotNull(target);
    this.method = method;
    method.setAccessible(true);

    this.executor = bus.executor();
  }

  /** Dispatches {@code event} to this subscriber using the proper executor. */
  final void dispatchEvent(Object event) {
    executor.execute(
        () -> {
          try {
            invokeSubscriberMethod(event);
          } catch (InvocationTargetException e) {
            bus.handleSubscriberException(e.getCause(), context(event));
          }
        });
  }

