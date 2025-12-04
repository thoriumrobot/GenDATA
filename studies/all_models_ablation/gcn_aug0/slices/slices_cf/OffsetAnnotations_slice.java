    @Positive
  public static void OffsetAnnotationsReader() throws IOException {
    @Positive
    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
    @Positive
    char[] buffer = new char[10];
    // :: error: (argument)
    @Positive
    bufferedReader.read(buffer, 5, 7);
    @Positive
  }
