  public static void method() throws IOException {
    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));

    // :: error: (argument)
    bufferedReader.skip(-1);

    bufferedReader.skip(1);
  }
