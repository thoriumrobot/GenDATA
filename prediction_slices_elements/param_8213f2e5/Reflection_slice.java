// Source-based slice around line 79
// Method: <com.google.common.reflect.Reflection: T newProxy(Class,InvocationHandler)>

  /**
   * Returns a proxy instance that implements {@code interfaceType} by dispatching method
   * invocations to {@code handler}. The class loader of {@code interfaceType} will be used to
   * define the proxy class. To implement multiple interfaces or specify a class loader, use {@link
   * Proxy#newProxyInstance}.
   *
   * @throws IllegalArgumentException if {@code interfaceType} does not specify the type of a Java
   *     interface
   */
  public static <T> T newProxy(Class<T> interfaceType, InvocationHandler handler) {
    checkNotNull(handler);
    checkArgument(interfaceType.isInterface(), "%s is not an interface", interfaceType);
    Object object =
        Proxy.newProxyInstance(
            interfaceType.getClassLoader(), new Class<?>[] {interfaceType}, handler);
    return interfaceType.cast(object);
  }

  private Reflection() {}
}
