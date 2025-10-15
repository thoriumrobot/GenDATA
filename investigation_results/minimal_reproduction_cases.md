# Minimal Reproduction Cases

## Phase 3: Isolating Issues with Minimal Test Cases

### Purpose
Create minimal, isolated test cases that demonstrate each type of failure without the complexity of the full test suite. These cases will help verify fixes and understand the exact behavior.

### Case 1: Variable Declaration Order Bug (CRITICAL)

#### Problem
Loop conversion places condition checks before variable declarations, causing undefined variable references.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }
}
```

#### Expected Transformation
```java
public class TestClass {
    public void test() {
        while (true) {
            int i = 0;           // ← Variable declaration first
            if (!(i < 10)) {     // ← Then condition check
                break;
            }
            System.out.println(i);
            i++;
        }
    }
}
```

#### Actual (Broken) Transformation
```java
public class TestClass {
    public void test() {
        while (true) {
            if (!i < 10) {       // ← ERROR: i is undefined here
                break;
            }
            int i = 0;           // ← Variable declared after usage
            System.out.println(i);
            i++;
        }
    }
}
```

#### Compilation Error
```
error: cannot find symbol
    if (!i < 10) {
         ^
  symbol:   variable i
  location: class TestClass
```

### Case 2: Multiple Variable References Bug (CRITICAL)

#### Problem
Complex for loops with multiple variables fail with multiple undefined variable references.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        for (int i = 0, j = 0; i < 10 && j < 5; i++, j++) {
            System.out.println(i + j);
        }
    }
}
```

#### Expected Transformation
```java
public class TestClass {
    public void test() {
        while (true) {
            int i = 0, j = 0;    // ← Multiple variables declared first
            if (!(i < 10 && j < 5)) {  // ← Then condition check
                break;
            }
            System.out.println(i + j);
            i++;
            j++;
        }
    }
}
```

#### Actual (Broken) Transformation
```java
public class TestClass {
    public void test() {
        while (true) {
            if (!i < 10 && j < 5) {  // ← ERROR: both i and j undefined
                break;
            }
            int i = 0, j = 0;    // ← Variables declared after usage
            System.out.println(i + j);
            i++;
            j++;
        }
    }
}
```

#### Compilation Errors
```
error: cannot find symbol
    if (!i < 10 && j < 5) {
         ^
  symbol:   variable i
  location: class TestClass

error: cannot find symbol
    if (!i < 10 && j < 5) {
                ^
  symbol:   variable j
  location: class TestClass
```

### Case 3: Guard Reversal on Loop Conditions Bug (HIGH)

#### Problem
Guard reversal is applied to if statements inside converted while loops, creating invalid conditions.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        for (int i = 0; i < 10; i++) {
            if (i > 5) {
                break;
            }
            System.out.println(i);
        }
    }
}
```

#### Expected Behavior
Guard reversal should NOT be applied to if statements inside converted loops.

#### Actual (Broken) Behavior
```java
public class TestClass {
    public void test() {
        while (true) {
            if (!i < 10) {       // ← Loop condition (correct)
                break;
            }
            int i = 0;
            if (!(i > 5)) {      // ← Guard reversal applied incorrectly
                break;
            }
            System.out.println(i);
            i++;
        }
    }
}
```

### Case 4: Semantic Equivalence Failure (HIGH)

#### Problem
While-to-for conversions produce semantically different code.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        int i = 0;
        while (i < 10) {
            System.out.println(i);
            i++;
        }
    }
}
```

#### Expected Transformation
```java
public class TestClass {
    public void test() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }
}
```

#### Actual (Broken) Transformation
```java
public class TestClass {
    public void test() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
            i++;  // ← Extra increment, changing semantics
        }
    }
}
```

### Case 5: Infinite Loop Preservation Bug (MEDIUM)

#### Problem
Infinite while loops are incorrectly modified when they should be preserved.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        while (true) {
            System.out.println("infinite");
        }
    }
}
```

#### Expected Behavior
Infinite while loops should be preserved unchanged.

#### Actual (Broken) Behavior
The transformation incorrectly modifies the infinite loop structure.

### Case 6: Method Call in Condition Bug (MEDIUM)

#### Problem
For loops with method calls in conditions fail with undefined variable references.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public int getMax() { return 10; }
    
    public void test() {
        for (int i = 0; i < getMax(); i++) {
            System.out.println(i);
        }
    }
}
```

#### Expected Transformation
```java
public class TestClass {
    public int getMax() { return 10; }
    
    public void test() {
        while (true) {
            int i = 0;
            if (!(i < getMax())) {  // ← Method call preserved
                break;
            }
            System.out.println(i);
            i++;
        }
    }
}
```

#### Actual (Broken) Transformation
```java
public class TestClass {
    public int getMax() { return 10; }
    
    public void test() {
        while (true) {
            if (!i < getMax()) {    // ← ERROR: i undefined, method call preserved
                break;
            }
            int i = 0;
            System.out.println(i);
            i++;
        }
    }
}
```

### Case 7: Array Access in Condition Bug (MEDIUM)

#### Problem
For loops with array access in conditions fail with undefined variable references.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        int[] arr = {1, 2, 3, 4, 5};
        for (int i = 0; i < arr.length; i++) {
            System.out.println(arr[i]);
        }
    }
}
```

#### Expected Transformation
```java
public class TestClass {
    public void test() {
        int[] arr = {1, 2, 3, 4, 5};
        while (true) {
            int i = 0;
            if (!(i < arr.length)) {  // ← Array access preserved
                break;
            }
            System.out.println(arr[i]);
            i++;
        }
    }
}
```

#### Actual (Broken) Transformation
```java
public class TestClass {
    public void test() {
        int[] arr = {1, 2, 3, 4, 5};
        while (true) {
            if (!i < arr.length) {    // ← ERROR: i undefined, array access preserved
                break;
            }
            int i = 0;
            System.out.println(arr[i]);
            i++;
        }
    }
}
```

### Case 8: Nested Loop Bug (MEDIUM)

#### Problem
Nested for loops fail with undefined variable references in outer loop conditions.

#### Minimal Test Case
```java
// Original code
public class TestClass {
    public void test() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.println(i + j);
            }
        }
    }
}
```

#### Expected Transformation
```java
public class TestClass {
    public void test() {
        while (true) {
            int i = 0;
            if (!(i < 3)) {
                break;
            }
            while (true) {
                int j = 0;
                if (!(j < 3)) {
                    break;
                }
                System.out.println(i + j);
                j++;
            }
            i++;
        }
    }
}
```

#### Actual (Broken) Transformation
```java
public class TestClass {
    public void test() {
        while (true) {
            if (!i < 3) {        // ← ERROR: i undefined
                break;
            }
            int i = 0;
            while (true) {
                int j = 0;
                if (!(j < 3)) {
                    break;
                }
                System.out.println(i + j);
                j++;
            }
            i++;
        }
    }
}
```

## Summary of Issues

### Critical Issues (Must Fix)
1. **Variable Declaration Order**: Variables declared after condition checks
2. **Multiple Variable References**: Multiple undefined variables in complex conditions

### High Priority Issues (Should Fix)
3. **Guard Reversal on Loop Conditions**: Incorrect application of guard reversal
4. **Semantic Equivalence Failures**: While-to-for conversions change meaning

### Medium Priority Issues (Could Fix)
5. **Infinite Loop Preservation**: Infinite loops incorrectly modified
6. **Method Call in Condition**: Method calls preserved but variables undefined
7. **Array Access in Condition**: Array access preserved but variables undefined
8. **Nested Loop Handling**: Outer loop variables undefined in nested scenarios

### Root Cause Analysis
All issues stem from the same fundamental problem: **the `convertForToWhile` method places condition checks before variable declarations**, creating undefined variable references throughout the converted code.

### Fix Strategy
1. **Fix variable declaration order** in `convertForToWhile` method
2. **Add context checking** to prevent guard reversal on loop conditions
3. **Improve while-to-for conversion** logic for semantic equivalence
4. **Add special handling** for infinite loops and complex conditions

## Next Steps

1. **Phase 4**: Implement fixes for critical issues
2. **Phase 5**: Validate fixes using these minimal reproduction cases
3. **Phase 6**: Update test expectations and add comprehensive coverage
