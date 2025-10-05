    @Positive
  public T get(@IndexFor("this") int index) {
    @Positive
    return (T) delegate[index];
    @Positive
  }
