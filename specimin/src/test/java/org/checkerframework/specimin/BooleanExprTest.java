package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/** This test checks that arguments to boolean expressions always have the type "boolean". */
public class BooleanExprTest {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "booleanexpr",
        new String[] {"com/example/Simple.java"},
        new String[] {"com.example.Simple#bar(Foo)"});
  }
}
