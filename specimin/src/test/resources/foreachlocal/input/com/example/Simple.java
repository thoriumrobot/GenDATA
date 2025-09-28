package com.example;

import java.util.List;

final class Simple {

    private static List<Foo> foos = null;

    // Target method. The goal of this test is to check that f is added to the local variable
    // scope, and not considered a field of the (non-existent) superclass.
    public static void bar() {
        for (Foo f : foos) {
            if (f instanceof Bar) {
                f.doSomething();
            }
        }
    }
}
